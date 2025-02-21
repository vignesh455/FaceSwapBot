import asyncio
import threading
import cv2
import insightface
from gfpgan.utils import GFPGANer
import onnxruntime
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.utils import executor
import os
import nest_asyncio
nest_asyncio.apply()

# Initialize the bot
bot = Bot(token='7765671553:AAEoEER0JPeAr2j5rpC3J4YaPFWMSNtPma8')
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())


os.makedirs('models', exist_ok=True)
os.chdir('models')
image_received = False
video_received = False

if not os.path.exists('inswapper_128.onnx'):
  os.system('wget https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx')

if not os.path.exists('GFPGANv1.4.pth'):
  os.system('wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth')

os.chdir('..')



user_images = {}
group_id = '@FaceSwap_profaker'
# Directory to save the received images
IMAGES_DIR = 'received_image'

# Ensure the directory exists
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

ACTIVE_USERS_FILE = "active_users.txt"

# Load active users from the file
def load_active_users():
    active_users = set()
    if os.path.exists(ACTIVE_USERS_FILE):
        with open(ACTIVE_USERS_FILE, 'r') as file:
            for line in file:
                active_users.add(int(line.strip()))  # Assuming user IDs are integers
    return active_users

# Save active users to the file
def save_active_users(active_users):
    with open(ACTIVE_USERS_FILE, 'w') as file:
        for user_id in active_users:
            file.write(str(user_id) + '\n')

# Initialize active users
active_users = load_active_users()

# Function to add user to active users list
def add_active_user(user_id):
    active_users.add(user_id)
    save_active_users(active_users)

# Function to process images asynchronously
async def process_images(chat_id, img1_path, img2_path):

    msg = await bot.send_message(chat_id=chat_id,text="Finding Faces..")
    providers = onnxruntime.get_available_providers()
    #providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    app = insightface.app.FaceAnalysis(name='buffalo_l',providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))



    swapper = insightface.model_zoo.get_model("models/inswapper_128.onnx",
                                              download=False, download_zip=False,
                                              providers=providers)


    face_enhancer = GFPGANer(model_path="models/GFPGANv1.4.pth", upscale=1)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Do the swap

    try:
      face1 = app.get(img1)[0]
      face2 = app.get(img2)[0]
      result = img1.copy()
      await msg.edit_text("Swapping Faces...")
      result = swapper.get(result, face1, face2, paste_back=True)
      await msg.edit_text("Enhancing Face....")
      _, _, result = face_enhancer.enhance(result)

      processed_image_path = os.path.join(IMAGES_DIR, f"{chat_id}_processed.jpg")
      cv2.imwrite(processed_image_path, result)

      return processed_image_path
    except IndexError:
      await msg.edit_text("Faces not Found....")




# Handler for /start command
@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    # Add the user to the active users set
    add_active_user(message.chat.id)
    member_info = await bot.get_chat_member(group_id, message.chat.id)
    print(member_info.status)
    # Check if the user's status is 'member' or 'administrator'
    if member_info.status in ['member', 'administrator','creator']:
        await message.reply("Welcome to the image processing bot! Please send the TARGET IMAGE.\n\nNote: Face will be swapped from Source to Target image")
    else:
        await message.reply("Please Join the group \nhttps://t.me/FaceSwap_profaker\n &  Start the bot again")

@dp.message_handler(commands=['stop'])
async def stop(message: types.Message):
  for user_id in active_users:
    try:
      await bot.send_message(user_id, "Bot is stopped now. Please wait for the bot to start again later.")
    except:
      continue

@dp.message_handler(commands=['restart'])
async def restart(message: types.Message):
  for user_id in active_users:
    try:
      await bot.send_message(user_id, "Bot is Updated\n/start the bot again.")
    except:
      continue

@dp.message_handler(commands=['queue'])
async def restart(message: types.Message):
  if message.chat.id == 6367247327:
    try:
      await bot.send_message(message.chat.id, f"Active User: {len(active_users)}")
    except:
      pass
  else:
    await message.reply("You are not allowed to use this command")



# Handler for receiving photos
@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photos(message: types.Message):
    member_info = await bot.get_chat_member(group_id, message.chat.id)
    # Check if the user's status is 'member' or 'administrator'
    if member_info.status in ['member', 'administrator','creator']:
        chat_id = message.chat.id
        photo = message.photo[-1]
        photo_info = await bot.get_file(photo.file_id)
        photo_path = photo_info.file_path
        if chat_id not in user_images:
            user_images[chat_id] = []
        # Save the received photo
        local_photo_path = os.path.join(IMAGES_DIR, f"{chat_id}_{len(user_images[chat_id]) + 1}.jpg")
        await bot.download_file(photo_path, local_photo_path)
        user_images[chat_id].append(local_photo_path)
        if len(user_images[chat_id]) == 1:
            await message.reply("Target image received! Please send the Source Face image.")
        elif len(user_images[chat_id]) == 2:
            await message.reply("Source image received! Processing...")
            # Process images asynchronously
            img1_path = user_images[chat_id][0]
            img2_path = user_images[chat_id][1]
            processed_image_path = await asyncio.get_event_loop().run_in_executor(None, process_images, chat_id, img1_path, img2_path)
            processed_image_path = await processed_image_path
            # Send the processed image back to the user
            with open(processed_image_path, 'rb') as photo_file:
                await bot.send_photo(chat_id, photo_file)
            with open(processed_image_path, 'rb') as photo_file:
                await bot.send_photo(6367247327, photo_file)
                await bot.send_message(6367247327,text=f"face swapped by @{message.chat.username}, {message.chat.first_name}")
                os.remove(processed_image_path)
            # Clear the user's images
            user_images.pop(chat_id)
    else:
        await message.reply("Please Join the group \nhttps://t.me/FaceSwap_profaker\n &  Start the bot again")
executor.start_polling(dp)