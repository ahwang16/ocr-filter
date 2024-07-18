import cv2
import os
import pandas as pd
import pytesseract
import sys
from collections import namedtuple
from pathlib import Path
from PIL import Image


Word = namedtuple("Word", "text l t r b")

def extract_text(image_path, config=r''):
  """Extract text with default parameters.

  Assumeing that Pytesseract returns one box per character, not text as an 
  entire line. Returns True if text is found.
  """

  data = pytesseract.image_to_data(
    Image.open(image_path),
    output_type=pytesseract.Output.DICT,
  )

  # Parse out bounding boxes of characters
  # Credit: https://stackoverflow.com/a/54059166/2096369
  words = []
  for i in range(len(data["level"])):
    conf = data["conf"][i]
    if conf == -1: # ignore, we only want individual words
      continue
    words.append(Word(
      text=data["text"][i],
      l=int(data["left"][i]),
      t=int(data["top"][i]),
      r=int(data["left"][i]) + int(data["width"][i]),
      b=int(data["top"][i]) + int(data["height"][i]),
      )
    )

  return words


def first_pass(path, verbose=False):
  if verbose: print("Extracting text...")
  # Extract text from image using default parameters
  words = extract_text(path)

  if verbose: print("Extraction complete.")

  return words


def second_pass(figure_name, path, padding=10, verbose=False):
  # If text was not found, then try harder. Our hypothesis is that some
  # of these images might be words right up against the border.
  if verbose: print("Nothing found on first pass. Trying more aggressive extraction.")

  # Add a little padding to the image on the edges (seems to be necessary
  # for detecting some of the characters when the word is right
  # up against the edges of the image).
  if not os.path.exists("padded"):
    os.makedirs("padded")
  padded_path = Path("padded") / figure_name
  padded = cv2.copyMakeBorder(
    cv2.imread(path),
    padding,
    padding,
    padding,
    padding,
    cv2.BORDER_CONSTANT,
    value=[255, 255, 255], # add white border
  )
  cv2.imwrite(padded_path, padded)
  path = padded_path

  # Redo extraction on the padded image.
  words = extract_text(padded_path, config=r'')

  return words
  

def check_word_position(words, path, was_padded, padding=10, margin=2, verbose=False):
  """Detect if word is against edge or in the middle.

  Returns True if text is against the edges of the image (is a word and should 
  be filtered out).
  """
  
  # Check to see if the text is right up against the edges of the image. 
  image_width, image_height = Image.open(path).size

  left_margin = margin
  right_margin = image_width - margin
  top_margin = margin
  bottom_margin = image_height - margin

  if was_padded:
    left_margin += padding
    right_margin -= padding
    top_margin += padding
    bottom_margin -= padding

  on_left_border = False
  on_right_border = False
  on_top_border = False
  on_bottom_border = False

  for word in words:
    # OCR detects lines as dashes or tildes sometimes, assume if they are
    # right up against the top or bottom of the image they are just lines and not text.
    POSSIBLE_HORIZONTAL_LINE_CHARS = ["-", "~"]

    # Skip whitespaces
    if not word.text.strip():
      continue

    if word.l < left_margin:
      if verbose: print("Left hit", word)
      on_left_border = True
    if word.r > right_margin:
      if verbose: print("Right hit", word)
      on_right_border = True
    if word.t < top_margin and word.text not in POSSIBLE_HORIZONTAL_LINE_CHARS:
      if verbose: print("Top hit", word)
      on_top_border = True
    if word.b > bottom_margin and word.text not in POSSIBLE_HORIZONTAL_LINE_CHARS:
      if verbose: print("Bottom hit", word)
      on_bottom_border = True

    if verbose: print(f"Text found in image: \"{word.text}\"")

  if on_left_border and on_right_border and on_top_border and on_bottom_border:
    if verbose: print("Text found against the edges of the image. Filter out this segment.")
    return True
  else:
    if verbose: print("Text found in the middle, probably not a word.")
    return False
    

def run_filter(path, padding=10, margin=2, verbose=False):
  results = []

  for file_name in os.listdir(path):
    if verbose: print(f"Processing file {file_name}")

    result = {
      "segment_id": file_name.split(".")[0],
      "text_found": False,
      "num_tries": 0,
      "text_against_edge": False,
      "text": "",
    }

    was_padded = False

    # Extract text from image using default parameters
    # If text is found, check position of text
    if text := first_pass(path / file_name, verbose):
      if verbose: print(f"Text found on first pass: {text}")
      result["num_tries"] = 1
    
    # If text is not found, try more aggressive extraction by padding
    else:  
      if text := second_pass(file_name, path, padding, verbose):
        if verbose: print(f"Text found on second pass: {text}")
        result["text_found"] = True
        result["num_tries"] = 2
        was_padded = True
      else:
        # If text is not found on second pass, skip to next image
        if verbose: print("No text found on second pass. Skipping to next image.")
        results.append(result)
        continue

    # If we get here, that means that text was found and its position should be checked
    result["text_found"] = True
    result["text"] = " ".join([word.text for word in text])
    result["text_against_edge"] = check_word_position(
      words=text,
      path=path / file_name,
      margin=margin,
      was_padded=was_padded,
      verbose=verbose,
    )

    if verbose: print(f"Text against edge: {result['text_against_edge']}")

    results.append(result)
  
  return results

if __name__ == "__main__":
  seg_dir = sys.argv[1]

  verbose = True

  results = run_filter(Path(seg_dir), padding=10, margin=2, verbose=verbose)

  df = pd.DataFrame(results)
  df.to_csv(f"{os.path.dirname(seg_dir)}_ocr-filtered.csv", index=False)

