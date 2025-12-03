import cv2
import os
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Your bot token
BOT_TOKEN = "Your telegram bot token"

# Function to process the image
def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using histogram equalization
    contrast_image = cv2.equalizeHist(gray_image)

    # Increase whites and darken blacks using contrast stretching
    min_val, max_val = gray_image.min(), gray_image.max()
    stretched_image = ((gray_image - min_val) / (max_val - min_val) * 255).astype('uint8')

    # Apply adaptive thresholding
    adaptive_threshold_image = cv2.adaptiveThreshold(
        stretched_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 9
    )

    # Save the processed image
    output_path = "processed_image.jpg"
    cv2.imwrite(output_path, adaptive_threshold_image)

    return output_path

# Command handler for the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Send me an image, and I'll process it for you!")

# Message handler for images
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo = update.message.photo[-1]  # Get the highest resolution photo
    photo_file = await photo.get_file()
    photo_path = "input_image.jpg"
    await photo_file.download_to_drive(photo_path)

    # Process the image
    processed_image_path = process_image(photo_path)

    # Send the processed image back to the user
    with open(processed_image_path, "rb") as f:
        await update.message.reply_photo(f)

    # Clean up temporary files
    os.remove(photo_path)
    os.remove(processed_image_path)

def main():
    # Initialize the bot application
    application = Application.builder().token(BOT_TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Start the bot
    # Use `application.run_polling()` directly if already inside an event loop
    asyncio.get_event_loop().run_until_complete(application.run_polling())

if __name__ == "__main__":
    main()
