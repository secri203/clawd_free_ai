@echo off
echo 正在查找监听9999端口的进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :9999') do (
    echo 正在关闭进程ID: %%a
    taskkill /F /PID %%a
)

echo 等待1秒确保进程完全关闭...
timeout /t 1 /nobreak > nul

echo 启动新进程...
start python yuanbao_openai_api.py
echo 重启完成！ 
