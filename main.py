import discord
from discord.ext import commands
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)


def get_class(image, model, classes):
  np.set_printoptions(suppress=True)
  
  model = load_model(model, compile=False)
 
  class_names = open(classes, "r").readlines()
  
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

  image = Image.open(image).convert("RGB")
  
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
  
  image_array = np.asarray(image)
  
  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
 
  data[0] = normalized_image_array
  
  prediction = model.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]

  print("Class:", class_name[2:], end="")
  print("Confidence Score:", confidence_score)
  
  return index

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hi! I am a bot {bot.user}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)


@bot.command()
async def test(ctx):
    files = ctx.message.attachments 
    await ctx.send(files)
    await ctx.send(files[0].filename )
    await ctx.send(files[0].url)
    syf = files[0].filename.split(".")[-1]
    await files[0].save(f"file.{syf}" )


@bot.command()
async def predict(ctx):
    files = ctx.message.attachments 
    syf = files[0].filename.split(".")[-1]
    
    if syf !='jpg' and syf !='jpeg' and syf !='png':
        await ctx.send("Неверное расширение")
        return
    
    await files[0].save(f"file.{syf}" )
    indx = get_class( f"file.{syf}",
                     "keras_model.h5",
                      "labels.txt" )
    
    if indx == 0:
        await ctx.send("Эта ягода спелая")
        await ctx.send("Можно собрать")
    else:
         await ctx.send("Эта ягода еще недоспела")
         await ctx.send("Можно собрать через несколько дней")


bot.run("")