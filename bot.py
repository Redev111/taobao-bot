import os
import logging
import time
import asyncio
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackQueryHandler, ContextTypes, filters
from PIL import Image
import numpy as np
import easyocr
import requests

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Проверяем доступность GPU
try:
    import torch
    use_gpu = torch.cuda.is_available()
except ImportError:
    use_gpu = False

# Инициализация EasyOCR для китайского языка
reader = easyocr.Reader(['ch_sim'], gpu=use_gpu)

# -------------------------
# Глобальные данные
# -------------------------

pending_data = {}  # {user_id: {"texts": [...], "task": Task, "start_time": ...}}

# -------------------------
# Функции очистки и форматирования
# -------------------------

def clean_text(text):
    """Очищает текст от лишних символов и форматирует"""
    replacements = {
        "**": "",
        "*": "",
        "- ": "",
        "• ": "",
        "1.": "",
        "2.": "",
        "3.": "",
        "4.": "",
        "5.": "",
        "Название товара": "🛒 ",
        "Основные характеристики": "\n⚙️ ",
        "Ценовые варианты": "\n💰 ",
        "Преимущества": "\n✅ ",
        "Отзывы": "\n🌟 ",
        "Дополнительная информация": "\nℹ️ "
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()

# -------------------------
# Обработчики команд и фото
# -------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📷 Начать анализ", callback_data="start_analysis")],
        [InlineKeyboardButton("🔁 Повторить", callback_data="repeat_analysis")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("👋 Привет! Я бот для анализа товаров с Taobao.\n\n"
                                    "Отправьте мне несколько скриншоты страницы товара, и я:\n"
                                    "🔹 Объединю информацию\n"
                                    "🔹 Переведу на русский\n"
                                    "🔹 Верну структурированный результат с ценами в юанях ¥", 
                                    reply_markup=reply_markup)


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "start_analysis":
        await query.edit_message_text("📸 Теперь отправьте скриншоты с Taobao. Как только закончите — подождите 10 секунд.")
    elif query.data == "repeat_analysis":
        await query.edit_message_text("🔄 Напишите /start, чтобы начать новый анализ.")


async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info(f"Получено фото от пользователя {user_id}")

    try:
        # Скачиваем изображение
        photo_file = await update.message.photo[-1].get_file()
        image_path = await photo_file.download_to_drive()

        # Открываем как изображение
        image = Image.open(image_path)
        image_np = np.array(image)

        # Распознаём текст
        results = reader.readtext(image_np)
        chinese_text = "\n".join([res[1] for res in results])

        if not chinese_text:
            await update.message.reply_text("❌ Не удалось распознать текст на этом изображении.")
            return

        # Сохраняем текст
        if user_id not in pending_data:
            pending_data[user_id] = {
                "texts": [],
                "task": None,
                "start_time": time.time()
            }

        pending_data[user_id]["texts"].append(chinese_text)
        count = len(pending_data[user_id]["texts"])
        await update.message.reply_text(f"🧠 Текст с {count}-го скриншота успешно распознан.")

        # Отменяем предыдущую задачу, если она есть
        if pending_data[user_id]["task"]:
            pending_data[user_id]["task"].cancel()

        # Запускаем новую задачу с задержкой
        pending_data[user_id]["task"] = context.application.create_task(
            delayed_finalize(user_id, context)
        )

    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        await update.message.reply_text("❌ Произошла ошибка при обработке изображения.")


async def delayed_finalize(user_id, context):
    try:
        # Ждём 10 секунд после последнего скриншота
        await asyncio.sleep(10)

        # Проверяем, нет ли новых скриншотов за это время
        data = pending_data.get(user_id)
        if not data or not data["texts"]:
            return

        # Выполняем обработку
        await finalize_auto(user_id, context)

    except asyncio.CancelledError:
        # Если задача была отменена — ничего не делаем
        pass


async def finalize_auto(user_id, context):
    logger.info(f"Авто-обработка для пользователя {user_id}")
    data = pending_data.get(user_id)
    if not data or not data["texts"]:
        return

    total_text = "\n\n".join(data["texts"])
    elapsed = round(time.time() - data["start_time"], 1)

    try:
        await context.bot.send_message(chat_id=user_id, text="🧠 Анализирую информацию со всех скриншотов...")

        prompt = f"""
        Ты профессиональный копирайтер для интернет-магазина. 
        На основе предоставленного текста с Taobao создай качественное описание товара на русском языке.

        Требования:
        1. Только русский язык, без китайских символов
        2. Цельный текст без маркированных списков
        3. Естественный стиль для объявления
        4. Включи все важные детали:
           - Название товара
           - Материал и размеры
           - Все варианты конфигураций и цены (указывай в юанях ¥)
           - Особенности и преимущества
           - Информацию о доставке
           - Скидки и акции
           - Отзывы (если есть в исходнике)

        Избегай:
        - Специальных символов (** /* и т.д.)
        - Нумерованных списков
        - Технических пометок

        Вот текст для анализа:
        {total_text}
        """

        headers = {
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json"
        }

        data_req = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }

        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions", 
            headers=headers,
            json=data_req,
            timeout=60
        )

        if response.status_code == 200:
            raw_text = response.json()['choices'][0]['message']['content']
            cleaned_text = clean_text(raw_text)
            final_text = f"{cleaned_text}\n\n——————————————\nℹ️ Чтобы начать новый анализ, отправьте новые скриншоты."

            await context.bot.send_message(chat_id=user_id, text=f"✅ Обработка завершена за {elapsed} секунд!")
            await context.bot.send_message(chat_id=user_id, text=final_text)
        else:
            error_msg = f"Ошибка API: {response.text}"
            logger.error(error_msg)
            await context.bot.send_message(chat_id=user_id, text="❌ Ошибка при обработке запроса к нейросети.")

        # Очистка данных
        pending_data.pop(user_id, None)

    except Exception as e:
        logger.error(f"Ошибка авто-обработки: {e}")
        await context.bot.send_message(chat_id=user_id, text="❌ Произошла ошибка при финальной обработке.")


# -------------------------
# Автоперезапуск бота
# -------------------------

def run_bot():
    while True:
        try:
            print("🔄 Запускаю бота...")
            application = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()

            # Регистрируем обработчики
            application.add_handler(CommandHandler("start", start))
            application.add_handler(CallbackQueryHandler(button_handler))
            application.add_handler(MessageHandler(filters.PHOTO, process_image))

            # Запускаем бота
            application.run_polling()
        except Exception as e:
            logger.error(f"Бот упал: {e}. Перезапускаю через 10 секунд...", exc_info=True)
            time.sleep(10)


# -------------------------
# Запуск бота
# -------------------------

if __name__ == "__main__":
    print("🚀 Бот запущен!")
    run_bot()