import subprocess
from watchfiles import watch

print("👀 Слежу за изменениями в bot.py...")

for changes in watch("bot.py"):
    print("🔄 Обнаружены изменения! Перезапускаю бота...")
    subprocess.run(["python", "bot.py"])