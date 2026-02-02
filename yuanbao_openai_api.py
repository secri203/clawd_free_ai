from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union, Generator
import uvicorn
import json
import requests
import urllib3
import time
import socket
import logging
import re
import os
import uuid

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI()

# 自定义异常处理器，提供更详细的验证错误信息
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        error_detail = {
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"]
        }
        errors.append(error_detail)
    
    logger.error(f"请求验证错误: {errors}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "请求参数验证失败",
            "errors": errors,
            "message": "请检查请求格式是否正确，确保包含所有必填字段"
        }
    )

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 定义支持的模型列表
SUPPORTED_MODELS = [
    {
        "name": "deepseek_v3",
        "modified_at": "2024-04-25T00:00:00.000000Z",
        "size": 0,
        "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    {
        "name": "deepseek_r1",
        "modified_at": "2024-04-25T00:00:00.000000Z",
        "size": 0,
        "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    {
        "name": "deepseek_public_v3",
        "modified_at": "2024-04-25T00:00:00.000000Z",
        "size": 0,
        "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    {
        "name": "deepseek_public_r1",
        "modified_at": "2024-04-25T00:00:00.000000Z",
        "size": 0,
        "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    }
]

# 定义模型到 chatModelId 的映射
MODEL_TO_CHAT_ID = {
    "deepseek_v3": "deep_seek_v3",
    "deepseek_r1": "deep_seek",
    "deepseek_public_v3": "deep_seek_v3",
    "deepseek_public_r1": "deep_seek"
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yuanbao_api.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

logger = logging.getLogger(__name__)

# 添加测试日志
logger.info("=== API服务启动 ===")
logger.info(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

class Message(BaseModel):
    role: str
    content: Optional[Any] = None

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None

class ChatCompletionRequest(BaseModel):
    model: str = "deepseek_v3"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: dict

class ChatCompletionMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

# 读取模型配置（所有模型共用同一个x_token）
def load_model_sessions():
    sessions = {}
    x_token = None
    try:
        # 获取脚本所在目录的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'yuanbao_model_sessions.txt')
        logger.info(f"正在读取配置文件: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 新格式：x_token:token值
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key, value = parts
                        if key.strip().lower() == 'x_token':
                            x_token = value.strip()
                            logger.info(f"加载 x_token: {x_token[:20]}...")
                            break

        # 如果成功读取到 x_token，为所有支持的模型创建配置
        if x_token:
            for model_name in MODEL_TO_CHAT_ID.keys():
                sessions[model_name] = {
                    'x_token': x_token
                }
                logger.info(f"为模型 {model_name} 配置 x_token")
            logger.info(f"成功为 {len(sessions)} 个模型配置 x_token")
        else:
            logger.error("配置文件中未找到 x_token")
    except Exception as e:
        logger.error(f"读取模型会话配置失败: {str(e)}")
    return sessions

# 全局变量存储会话ID映射
MODEL_SESSIONS = load_model_sessions()

# 存储每个模型的对话ID，每次请求后更新
MODEL_CONVERSATION_IDS = {}

def create_conversation(x_token: str, model: str) -> str:
    """
    创建新的对话
    """
    url = "https://yuanbao.tencent.com/api/user/agent/conversation/v1/detail"
    
    headers = {
        "sec-ch-ua-platform": "Windows",
        "x-device-id": "78755567e30633de2280724c300013617a17",
        "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Microsoft Edge";v="144", "Microsoft Edge WebView2";v="144"',
        "x-token": x_token,
        "x-instance-id": "1",
        "x-device-name": "Administrator",
        "sec-ch-ua-mobile": "?0",
        "x-language": "zh-CN",
        "x-requested-with": "XMLHttpRequest",
        "accept": "application/json, text/plain, */*",
        "x-a153": "78755567e30633de2280724c300013617a17",
        "content-type": "application/json",
        "x-trid-channel": "",
        "x-operationsystem": "win",
        "x-channel": "7001",
        "x-id": "c03ce86d8f624ae1af7c1cbeea4161d7",
        "x-product": "bot",
        "x-a10": "HP-2Q523AV",
        "x-appversion": "2.55.0",
        "x-source": "web",
        "x-a19": "wifi",
        "x-os_version": "Windows_11_Desktop",
        "x-hy92": "8f95c0219a98bc38c16530540200000901a202",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0 app_version/2.55.0 os_version/10.0.22000 app_theme/system os_name/windows app_short_version/2.55.0 app_lang/zh-CN system_lang/zh-CN app_instance_id/2 product_id/TM_Product_App app_full_version/2.55.0.611 package_type/publish_release c_district/0 app/tencent_yuanbao",
        "x-a3": "78755567e30633de2280724c300013617a17",
        "x-hy93": "1931b0f29271003d9a98bc38c165305447cafa1cb9",
        "x-a9": "HP",
        "origin": "https://tencent.yuanbao",
        "sec-fetch-site": "cross-site",
        "sec-fetch-mode": "cors",
        "sec-fetch-dest": "empty",
        "referer": "https://tencent.yuanbao/",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "priority": "u=1, i"
    }
    
    # 生成新的conversationId (UUID格式)
    conversation_id = str(uuid.uuid4())
    
    payload = {
        "conversationId": conversation_id,
        "limit": 30,
        "offset": 0
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, verify=False)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"创建对话成功: {conversation_id}")
        
        # 保存对话ID
        MODEL_CONVERSATION_IDS[model] = conversation_id
        
        return conversation_id
    except Exception as e:
        logger.error(f"创建对话失败: {str(e)}")
        raise


def get_or_create_conversation(x_token: str, model: str, force_create: bool = False) -> str:
    """
    获取或创建对话ID
    如果该模型已经有对话ID，则复用；否则创建新的
    
    Args:
        x_token: 认证token
        model: 模型名称
        force_create: 是否强制创建新对话（用于对话失效后重建）
    """
    # 检查是否已有对话ID且不需要强制创建
    if not force_create and model in MODEL_CONVERSATION_IDS:
        conversation_id = MODEL_CONVERSATION_IDS[model]
        logger.info(f"复用现有对话ID: {conversation_id}")
        return conversation_id
    
    # 如果需要强制创建，先清除旧的对话ID
    if force_create and model in MODEL_CONVERSATION_IDS:
        old_id = MODEL_CONVERSATION_IDS[model]
        logger.info(f"强制创建新对话，清除旧对话ID: {old_id}")
        del MODEL_CONVERSATION_IDS[model]
    
    # 创建新的对话
    return create_conversation(x_token, model)

def parse_tool_call(response_text: str, has_tool_result_in_history: bool = False) -> Optional[dict]:
    """
    解析 AI 响应，检查是否是 tool call 格式
    返回 tool call 字典或 None
    
    Args:
        response_text: AI 的响应文本
        has_tool_result_in_history: 对话历史中是否已经有 tool 执行结果
    """
    # 如果已经有 tool 执行结果，不应该再返回 tool call（防止死循环）
    if has_tool_result_in_history:
        logger.info("对话历史中已有 tool 执行结果，跳过 tool call 检测")
        return None
    
    try:
        text = response_text.strip()
        
        # 方法1: 检查是否是 Markdown 代码块格式
        if '```' in text:
            # 提取代码块内容
            code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if code_block_match:
                text = code_block_match.group(1).strip()
                logger.info(f"从代码块提取的JSON: {text[:100]}")
        
        # 方法2: 从文本中提取 JSON 对象（处理前面有额外文本的情况）
        # 查找 { 开始和 } 结束的完整 JSON 对象
        json_match = re.search(r'\{[^{}]*"type"\s*:\s*"tool_call"[^{}]*\}', text, re.DOTALL)
        if not json_match:
            # 尝试更宽松的匹配，支持嵌套对象
            json_match = re.search(r'\{.*"type"\s*:\s*"tool_call".*\}', text, re.DOTALL)
        
        if json_match:
            text = json_match.group(0)
            logger.info(f"提取的JSON文本: {text[:100]}")
        
        # 尝试解析 JSON
        data = json.loads(text)
        
        # 检查是否是 tool call 格式
        if isinstance(data, dict) and data.get('type') == 'tool_call':
            logger.info(f"成功解析 tool call: {data}")
            return {
                'tool': data.get('tool', ''),
                'command': data.get('command', ''),
                'comment': data.get('comment', ''),
                'args': data.get('args', [])
            }
    except (json.JSONDecodeError, ValueError) as e:
        logger.debug(f"不是 tool call 格式: {str(e)}")
    
    return None

def clean_chinese_text(text: str) -> str:
    """清理中文文本，保持良好的格式和段落结构"""
    
    # 处理标题格式
    text = re.sub(r'(\d+\..*?架构)\s*', r'\1\n\n', text)
    
    # 处理步骤标记
    text = re.sub(r'([A-Z]-[^-]*?)-([^-]*?)-', r'\n\1-\2-\n', text)
    
    # 处理步骤内容，在逗号后添加换行
    text = re.sub(r'([^,]*?),((?:[^,]*?,)*[^,]*?)(?=\n|$)', lambda m: f"{m.group(1)}\n{m.group(2)}", text)
    
    # 处理连续的换行符
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 确保最后有一个换行符
    text = text.strip() + '\n'
    
    return text

def is_conversation_invalid_error(error_msg: str) -> bool:
    """
    判断错误是否是对话失效/不存在的错误
    """
    invalid_keywords = [
        "conversation",
        "not found",
        "not exist",
        "deleted",
        "invalid",
        "不存在",
        "已删除",
        "无效",
        "找不到",
        "404"
    ]
    error_lower = str(error_msg).lower()
    return any(keyword in error_lower for keyword in invalid_keywords)


def send_yuanbao_request_with_retry(prompt: str, stream: bool = False, model: str = "deepseek_v3", max_retries: int = 1) -> Union[str, Generator[str, None, None]]:
    """
    发送请求到元宝API，支持对话失效后自动重试
    
    Args:
        prompt: 提示词
        stream: 是否流式输出
        model: 模型名称
        max_retries: 最大重试次数（对话失效后重新创建对话的次数）
    """
    # 获取对应模型的配置
    model_config = MODEL_SESSIONS.get(model)
    if not model_config:
        raise ValueError(f"未找到模型 {model} 的配置")
    
    retry_count = 0
    force_create = False
    
    while retry_count <= max_retries:
        # 获取或创建对话ID
        try:
            conversation_id = get_or_create_conversation(model_config['x_token'], model, force_create=force_create)
            logger.info(f"使用对话ID: {conversation_id} (尝试 {retry_count + 1}/{max_retries + 1})")
        except Exception as e:
            logger.error(f"获取/创建对话失败: {str(e)}")
            raise
        
        url = f"https://yuanbao.tencent.com/api/chat/{conversation_id}"
        
        headers = {
            "sec-ch-ua-platform": "Windows",
            "sec-ch-ua": '"Microsoft Edge WebView2";v="135", "Chromium";v="135", "Not-A.Brand";v="8", "Microsoft Edge";v="135"',
            "x-token": model_config['x_token'],
            "x-instance-id": "1",
            "sec-ch-ua-mobile": "?0",
            "x-language": "zh-CN",
            "x-requested-with": "XMLHttpRequest",
            "content-type": "text/plain;charset=UTF-8",
            "x-operationsystem": "win",
            "x-channel": "7001",
            "chat_version": "v1",
            "x-id": "c03ce86d8f624ae1af7c1cbeea4161d7",
            "sidebarwidth": "204",
            "x-a10": "HP-2Q523AV",
            "x-product": "bot",
            "x-appversion": "1.10.0",
            "x-os_version": "Windows_11_Desktop",
            "x-source": "web",
            "x-hy92": "defb4cb89a98bc38c16530540200000c119419",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0 product_id/TM_Product_App app_theme/system app/tencent_yuanbao os_name/windows app_short_version/1.10.0 app_instance_id/2 c_district/0 system_lang/zh-CN os_version/10.0.22000 app_version/1.10.0 package_type/publish_release app_full_version/1.10.0.608 app_lang/zh-CN",
            "x-a3": "78755567e30633de2280724c300013617a17",
            "x-hy93": "1931b0f29271003d9a98bc38c165305447cafa1cb9",
            "x-a9": "HP",
            "accept": "*/*",
            "origin": "https://tencent.yuanbao",
            "sec-fetch-site": "cross-site",
            "sec-fetch-mode": "cors",
            "sec-fetch-dest": "empty",
            "referer": "https://tencent.yuanbao/",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "priority": "u=1, i"
        }
        
        payload = {
            "agentId": "naQivTmsDa",
            "displayPrompt": prompt,
            "supportFunctions": [""],
            "version": "v2",
            "docOpenid": "144115210554304601",
            "multimedia": [],
            "plugin": "Adaptive",
            "supportHint": 1,
            "displayPromptType": 1,
            "options": {
                "imageIntention": {
                    "needIntentionModel": True,
                    "backendUpdateFlag": 2,
                    "intentionStatus": True
                }
            },
            "model": "gpt_175B_0404",
            "chatModelId": MODEL_TO_CHAT_ID.get(model, "deep_seek_v3"),
            "prompt": prompt
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, verify=False, stream=True)
            response.raise_for_status()
            
            # 请求成功，处理响应
            if stream:
                return _handle_stream_response(response, model)
            else:
                return _handle_normal_response(response, model)
                
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            logger.error(f"HTTP错误: {error_msg}")
            
            # 检查是否是对话失效的错误
            if is_conversation_invalid_error(error_msg) and retry_count < max_retries:
                logger.warning(f"对话可能已失效，准备重新创建对话并重试...")
                force_create = True  # 下次循环强制创建新对话
                retry_count += 1
                continue
            else:
                # 其他错误或重试次数已用完，抛出异常
                raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"请求异常: {error_msg}")
            
            # 检查是否是对话失效的错误
            if is_conversation_invalid_error(error_msg) and retry_count < max_retries:
                logger.warning(f"对话可能已失效，准备重新创建对话并重试...")
                force_create = True  # 下次循环强制创建新对话
                retry_count += 1
                continue
            else:
                raise
    
    # 如果执行到这里，说明重试次数已用完
    raise Exception(f"请求失败，已重试 {max_retries} 次")


def _handle_stream_response(response, model: str):
    """处理流式响应"""
    def generate():
        full_response = []
        current_thought = []
        thinking_started = False
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                logger.info(f"原始响应行: {line}")
                
                # 跳过非JSON数据
                if line in ['status', 'text']:
                    continue
                    
                if line.startswith('data: '):
                    try:
                        data = line[6:]
                        if data:
                            # 跳过非JSON标记行
                            if data.startswith('[MSGINDEX:') or data.startswith('[TRACEID:') or data.startswith('[DONE]'):
                                logger.info(f"跳过标记行: {data}")
                                continue
                            json_data = json.loads(data)
                            # 处理思考过程
                            if json_data.get('type') == 'think':
                                msg = json_data.get('content', '')
                                if msg:
                                    if not thinking_started:
                                        # 第一次遇到思考内容时，发送思考开始标记
                                        chunk = {
                                            "id": f"chatcmpl-{str(hash(''.join(full_response)))}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": "deepseek_v3",
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {
                                                        "role": "assistant" if len(full_response) == 0 else None,
                                                        "content": "<think>\n"
                                                    },
                                                    "finish_reason": None
                                                }
                                            ]
                                        }
                                        thinking_started = True
                                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                    
                                    current_thought.append(msg)
                                    # 当遇到句子结束标记时，发送完整的思考内容
                                    if msg.strip() in ['。', '？', '！', '.', '?', '!']:
                                        thought_text = ''.join(current_thought)
                                        chunk = {
                                            "id": f"chatcmpl-{str(hash(''.join(full_response)))}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": "deepseek_v3",
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {
                                                        "content": thought_text + "\n"
                                                    },
                                                    "finish_reason": None
                                                }
                                            ]
                                        }
                                        current_thought = []
                                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            
                            # 处理普通文本消息
                            elif json_data.get('type') == 'text':
                                msg = json_data.get('msg', '')
                                if msg:
                                    # 如果之前有未完成的思考内容，先发送出去
                                    if current_thought:
                                        thought_text = ''.join(current_thought)
                                        chunk = {
                                            "id": f"chatcmpl-{str(hash(''.join(full_response)))}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": "deepseek_v3",
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {
                                                        "content": thought_text + "\n"
                                                    },
                                                    "finish_reason": None
                                                }
                                            ]
                                        }
                                        current_thought = []
                                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                    
                                    # 如果是第一个文本消息且之前有思考过程，添加思考结束标记和换行
                                    if thinking_started and len(full_response) == 0:
                                        chunk = {
                                            "id": f"chatcmpl-{str(hash(''.join(full_response)))}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": "deepseek_v3",
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {
                                                        "content": "</think>\n\n"
                                                    },
                                                    "finish_reason": None
                                                }
                                            ]
                                        }
                                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                    
                                    # 发送实际的文本消息
                                    chunk = {
                                        "id": f"chatcmpl-{str(hash(''.join(full_response)))}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": "deepseek_v3",
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {
                                                    "role": "assistant" if len(full_response) == 0 and not thinking_started else None,
                                                    "content": msg
                                                },
                                                "finish_reason": None
                                            }
                                        ]
                                    }
                                    full_response.append(msg)
                                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析错误: {str(e)}")
                        continue
        
        # 发送结束标记
        end_chunk = {
            "id": f"chatcmpl-{str(hash(''.join(full_response)))}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "deepseek_v3",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    
    return generate()


def _handle_normal_response(response, model: str):
    """处理普通（非流式）响应"""
    full_response = []
    current_thought = []  # 用于收集当前思考过程的词组
    thinking_paragraphs = []  # 用于存储完整的思考段落
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            logger.info(f"原始响应行: {line}")
            
            # 跳过非JSON数据
            if line in ['status', 'text']:
                continue
                
            if line.startswith('data: '):
                try:
                    data = line[6:]
                    if data:
                        # 跳过标记行
                        if data.startswith('[MSGINDEX:') or data.startswith('[TRACEID:') or data.startswith('[DONE]'):
                            logger.info(f"跳过标记行: {data}")
                            continue
                        
                        json_data = json.loads(data)
                        
                        # 处理思考过程
                        if json_data.get('type') == 'think':
                            content = json_data.get('content', '')
                            if content:
                                current_thought.append(content)
                                # 当遇到句子结束标记时，将当前思考段落保存
                                if content.strip() in ['。', '？', '！', '.', '?', '!']:
                                    thought_text = ''.join(current_thought)
                                    thinking_paragraphs.append(thought_text)
                                    current_thought = []
                        
                        # 处理普通文本消息
                        elif json_data.get('type') == 'text':
                            msg = json_data.get('msg', '')
                            if msg:
                                full_response.append(msg)
                                
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误: {str(e)}")
                    continue
    
    # 处理剩余的思考内容
    if current_thought:
        thinking_paragraphs.append(''.join(current_thought))
    
    # 组合最终响应
    if thinking_paragraphs:
        # 如果有思考过程，使用<think>标签包裹
        thinking_text = '\n'.join(thinking_paragraphs)
        response_text = f"<think>\n{thinking_text}\n</think>\n\n{''.join(full_response)}"
    else:
        response_text = ''.join(full_response)
    
    return response_text


def send_yuanbao_request(prompt: str, stream: bool = False, model: str = "deepseek_v3") -> Union[str, Generator[str, None, None]]:
    """
    发送请求到元宝API（兼容旧接口，内部调用带重试的版本）
    """
    return send_yuanbao_request_with_retry(prompt, stream=stream, model=model, max_retries=1)


async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # 记录完整的请求内容
        logger.info("\n=== 收到Dify请求 ===")
        logger.info(f"完整请求内容: {request.model_dump_json(indent=2)}")
        logger.info(f"当前模型对话ID缓存: {MODEL_CONVERSATION_IDS}")
        
        # 获取系统提示词
        system_message = next((msg.content for msg in request.messages if msg.role == "system"), None)
        logger.info(f"提取到的系统提示词: {system_message}")
        
        # 获取最后一条用户消息
        user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
        logger.info(f"提取到的用户消息: {user_message}")
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # 如果有系统提示词，使用指令格式包装
        if system_message:
            # 使用指令格式，让元宝明白这是系统指令
            user_message = f"[系统指令]\n{system_message}\n\n[用户输入]\n{user_message}\n\n请严格按照系统指令执行，不要返回问候语或其他无关内容。"
            logger.info(f"合并后的完整提示词: {user_message}")

        # 如果是流式请求
        if request.stream:
            return StreamingResponse(
                send_yuanbao_request(user_message, stream=True),
                media_type="text/event-stream"
            )

        # 非流式请求
        response_text = send_yuanbao_request(user_message)
        logger.info(f"元宝API返回的响应: {response_text}")
        logger.info("=== 请求处理完成 ===\n")

        # 构建OpenAI格式的响应
        response = ChatCompletionResponse(
            id="chatcmpl-" + str(hash(response_text)),
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(user_message),
                "completion_tokens": len(response_text),
                "total_tokens": len(user_message) + len(response_text)
            }
        )

        return response
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    try:
        try:
            response_text = send_yuanbao_request(request.prompt, model=request.model)
        except Exception as e:
            response_text = str(e)
            
        return {
            "model": request.model,
            "response": response_text,
            "done": True
        }
    except Exception as e:
        return {
            "model": request.model,
            "response": str(e),
            "done": True
        }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # 获取系统提示词
        system_message = next((msg.content for msg in request.messages if msg.role == "system"), None)
        logger.info(f"提取到的系统提示词: {system_message}")
        
        # 获取最后一条用户消息
        user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
        logger.info(f"提取到的用户消息: {user_message}")
        
        if not user_message:
            return {
                "model": request.model,
                "message": {
                    "role": "assistant",
                    "content": "No user message found"
                },
                "done": True
            }

        # 如果有系统提示词，将其添加到用户消息前
        if system_message:
            user_message = f"{system_message}\n\n用户问题：{user_message}"
            logger.info(f"合并后的完整提示词: {user_message}")

        try:
            response_text = send_yuanbao_request(user_message, model=request.model)
        except Exception as e:
            response_text = str(e)
            
        logger.info(f"元宝API返回的响应: {response_text}")

        return {
            "model": request.model,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "done": True
        }
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        return {
            "model": request.model,
            "message": {
                "role": "assistant",
                "content": str(e)
            },
            "done": True
        }

@app.get("/api/clear_conversations")
async def clear_conversations():
    """清除所有对话缓存，强制创建新对话"""
    global MODEL_CONVERSATION_IDS
    MODEL_CONVERSATION_IDS = {}
    logger.info("已清除所有对话缓存")
    return {"status": "ok", "message": "所有对话缓存已清除"}

@app.get("/api/tags")
async def get_models():
    """返回支持的模型列表"""
    return {"models": SUPPORTED_MODELS}

@app.get("/api/version")
async def version():
    return {
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Yuanbao API is running"}

@app.get("/v1/models")
async def list_models():
    """返回支持的模型列表（OpenAI兼容格式）"""
    return {
        "object": "list",
        "data": SUPPORTED_MODELS
    }

def get_ip():
    try:
        # 获取主机名
        hostname = socket.gethostname()
        # 获取IP地址
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except:
        return "获取IP失败"

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # 记录请求信息
    logger.info("\n=== 收到请求 ===")
    logger.info(f"请求方法: {request.method}")
    logger.info(f"请求路径: {request.url.path}")
    logger.info(f"完整URL: {request.url}")
    logger.info(f"查询参数: {dict(request.query_params)}")
    logger.info(f"请求头: {dict(request.headers)}")
    logger.info(f"客户端IP: {request.client.host if request.client else '未知'}")
    logger.info(f"用户代理: {request.headers.get('user-agent', '未知')}")
    
    # 记录请求体（所有方法都记录）
    try:
        body = await request.body()
        if body:
            logger.info(f"原始请求体: {body.decode('utf-8')}")
        # Reset the request body so the endpoint can read it
        request._body = body
    except Exception as e:
        logger.error(f"读取请求体时出错: {str(e)}")
    
    # 处理请求
    response = await call_next(request)
    
    # 记录响应信息
    process_time = time.time() - start_time
    logger.info(f"处理时间: {process_time:.2f}秒")
    logger.info(f"响应状态: {response.status_code}")
    logger.info(f"响应头: {dict(response.headers)}")
    logger.info("=== 请求处理完成 ===")
    
    return response

@app.post("/v1/chat/completions")
async def openai_chat_completion(request: ChatCompletionRequest):
    try:
        logger.info("\n=== 收到OpenAI兼容请求 ===")
        logger.info(f"完整请求内容: {request.model_dump_json(indent=2)}")
        
        # 构建完整的对话历史
        conversation_history = []
        has_tool_call_in_history = False
        has_tool_result_in_history = False
        
        for msg in request.messages:
            role = msg.role
            content = msg.content
            
            # 处理不同类型的content
            if isinstance(content, list):
                # 处理multimodal内容（数组）
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                content_text = ' '.join(text_parts)
            elif isinstance(content, str):
                content_text = content
            else:
                # 其他类型转换为字符串
                content_text = str(content)
            
            if content_text:
                if role == 'system':
                    conversation_history.append(f"System: {content_text}")
                elif role == 'user':
                    conversation_history.append(f"User: {content_text}")
                elif role == 'assistant':
                    conversation_history.append(f"Assistant: {content_text}")
                    # 检查 assistant 消息是否包含 tool_calls
                    if content_text and '"tool_calls"' in content_text:
                        has_tool_call_in_history = True
                elif role == 'tool':
                    # tool 角色的消息是工具执行结果
                    conversation_history.append(f"Tool执行结果: {content_text}")
                    has_tool_result_in_history = True
        
        # 确保有用户消息
        if not any('User:' in msg for msg in conversation_history):
            raise HTTPException(status_code=400, detail="No user message found")
        
        # 添加指令，防止死循环
        if has_tool_result_in_history:
            # 如果已经有 tool 执行结果，告诉 AI 这是执行结果，应该返回普通文本总结
            conversation_history.append("\n[System指令: 上面是工具执行的结果。请根据执行结果给用户一个友好的总结回复，不要再次执行相同的命令。]")
        
        # 构建完整提示
        user_message = '\n'.join(conversation_history)

        # 如果是流式请求
        if request.stream:
            # 对于流式请求，我们先收集完整响应，检查是否是 tool call
            # 使用列表缓存所有 chunk，避免重复请求
            async def stream_with_tool_call():
                # 缓存所有 chunk
                cached_chunks = []
                full_response_parts = []
                
                # 发送流式请求并缓存所有 chunk
                for chunk in send_yuanbao_request(user_message, stream=True, model=request.model):
                    cached_chunks.append(chunk)
                    
                    # 解析 chunk 获取内容
                    if chunk.startswith('data: ') and not chunk.startswith('data: [DONE]'):
                        try:
                            data = json.loads(chunk[6:])
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    full_response_parts.append(content)
                        except:
                            pass
                
                # 合并完整响应
                full_response_text = ''.join(full_response_parts)
                
                # 检查是否是 tool call（传入 has_tool_result_in_history 防止死循环）
                tool_call_data = parse_tool_call(full_response_text, has_tool_result_in_history)
                
                if tool_call_data:
                    # 是 tool call，返回单个包含 tool_calls 的 chunk
                    logger.info(f"流式响应检测到 tool call: {tool_call_data}")
                    tool_call_id = f"call_{str(hash(full_response_text))}"
                    
                    # 构造 function 参数
                    function_args = {
                        "command": tool_call_data.get('command', ''),
                        "args": tool_call_data.get('args', [])
                    }
                    
                    chunk = {
                        "id": f"chatcmpl-{str(hash(full_response_text))}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": tool_call_id,
                                            "type": "function",
                                            "function": {
                                                "name": tool_call_data.get('tool', 'exec'),
                                                "arguments": json.dumps(function_args, ensure_ascii=False)
                                            }
                                        }
                                    ]
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    
                    # 发送结束标记
                    end_chunk = {
                        "id": f"chatcmpl-{str(hash(full_response_text))}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls"
                            }
                        ]
                    }
                    yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                else:
                    # 普通文本响应，直接返回缓存的 chunks
                    logger.info(f"普通流式响应，返回 {len(cached_chunks)} 个缓存 chunk")
                    for chunk in cached_chunks:
                        yield chunk
            
            return StreamingResponse(
                stream_with_tool_call(),
                media_type="text/event-stream"
            )
        
        # 非流式请求
        try:
            response_text = send_yuanbao_request(user_message, model=request.model)
        except Exception as e:
            # 将错误信息作为正常响应返回
            response_text = str(e)
        
        # 检查是否是 tool call（传入 has_tool_result_in_history 防止死循环）
        tool_call_data = parse_tool_call(response_text, has_tool_result_in_history)
        
        if tool_call_data:
            # 是 tool call，构造 tool_calls 格式的响应
            logger.info(f"检测到 tool call: {tool_call_data}")
            tool_call_id = f"call_{str(hash(response_text))}"
            
            # 构造 function 参数
            function_args = {
                "command": tool_call_data.get('command', ''),
                "args": tool_call_data.get('args', [])
            }
            
            response_data = {
                "id": f"chatcmpl-{str(hash(response_text))}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call_data.get('tool', 'exec'),
                                        "arguments": json.dumps(function_args, ensure_ascii=False)
                                    }
                                }
                            ]
                        },
                        "finish_reason": "tool_calls"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_message),
                    "completion_tokens": len(response_text),
                    "total_tokens": len(user_message) + len(response_text)
                }
            }
        else:
            # 普通文本响应
            response_data = {
                "id": f"chatcmpl-{str(hash(response_text))}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_message),
                    "completion_tokens": len(response_text),
                    "total_tokens": len(user_message) + len(response_text)
                }
            }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        logger.error(f"错误类型: {type(e)}")
        logger.error(f"错误详情: {str(e)}")
        # 将错误信息作为正常响应返回
        user_message = request.messages[-1].content if request.messages else None
        response_data = {
            "id": f"chatcmpl-{str(hash(str(e)))}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(e)
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message) if user_message else 0,
                "completion_tokens": len(str(e)),
                "total_tokens": (len(user_message) if user_message else 0) + len(str(e))
            }
        }
        return JSONResponse(content=response_data)

@app.post("/v1/responses")
async def openai_responses(request: ChatCompletionRequest):
    try:
        logger.info("\n=== 收到OpenAI兼容responses请求 ===")
        logger.info(f"完整请求内容: {request.model_dump_json(indent=2)}")
        
        # 获取用户消息
        user_message = request.messages[-1].content if request.messages else None
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # 如果是流式请求
        if request.stream:
            return StreamingResponse(
                send_yuanbao_request(user_message, stream=True, model=request.model),
                media_type="text/event-stream"
            )
        
        # 非流式请求
        try:
            response_text = send_yuanbao_request(user_message, model=request.model)
        except Exception as e:
            # 将错误信息作为正常响应返回
            response_text = str(e)
        
        response_data = {
            "id": f"resp-{str(hash(response_text))}",
            "object": "response",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message),
                "completion_tokens": len(response_text),
                "total_tokens": len(user_message) + len(response_text)
            }
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"处理responses请求时发生错误: {str(e)}")
        logger.error(f"错误类型: {type(e)}")
        logger.error(f"错误详情: {str(e)}")
        # 将错误信息作为正常响应返回
        user_message = request.messages[-1].content if request.messages else None
        response_data = {
            "id": f"resp-{str(hash(str(e)))}",
            "object": "response",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(e)
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message) if user_message else 0,
                "completion_tokens": len(str(e)),
                "total_tokens": (len(user_message) if user_message else 0) + len(str(e))
            }
        }
        return JSONResponse(content=response_data)

if __name__ == "__main__":
    ip = get_ip()
    logger.info(f"服务器IP地址: {ip}")
    logger.info("服务即将启动...")
    
    logger.info("\n=== Yuanbao API Server ===")
    logger.info(f"Local IP address: {ip}")
    logger.info(f"Server will be available at:")
    logger.info(f"- Local: http://127.0.0.1:9999")
    logger.info(f"- Network: http://{ip}:9999")
    logger.info(f"- Docker: http://host.docker.internal:9999")
    logger.info("\nFor Dify in Docker, use either Network or Docker address")
    logger.info("You can test the server using:")
    logger.info(f"curl http://{ip}:9999/health")
    logger.info("===============================\n")
    
    logger.info("服务已启动，等待请求...")
    uvicorn.run(app, host="0.0.0.0", port=9999) 