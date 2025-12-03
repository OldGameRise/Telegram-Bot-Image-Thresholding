# Telegram Image Thresholding Bot

A lightweight Telegram bot that applies adaptive thresholding to images using OpenCV—perfect for turning photos into clean, high-contrast binary images.

## ✨ Features

- Accepts photos via Telegram  
- Converts to grayscale, enhances contrast, and applies adaptive thresholding  
- Returns the processed image instantly  
- Auto-cleans temporary files  

## 🛠️ Requirements

- Python 3.7+  
- Libraries: `python-telegram-bot`, `opencv-python`  

Install with:

```bash
pip install python-telegram-bot opencv-python
```

## ⚙️ Setup

1. Create a bot with [BotFather](https://t.me/BotFather) and get your token.  
2. Replace `"Your telegram bot token"` in the script with your actual token.  
3. Run the bot:

```bash
python TelegramCV2_threshholding.py
```

4. Send an image to your bot on Telegram!

## 📄 License

MIT License — free to use, modify, and share.
