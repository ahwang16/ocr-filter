from PIL import Image
import os
from collections import namedtuple

import pytesseract
import cv2
import sys


Word = namedtuple("Word", "text l t r b")


def extract_text(image_path, config=r''):
    " Probably returns one box per character, not text as an entire line. "

    data = pytesseract.image_to_data(
            Image.open(image_path),
            output_type=pytesseract.Output.DICT
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
            ))

    return words


if __name__ == "__main__":
    seg_dir = sys.argv[1]

    VERBOSE = False

    for file in os.listdir(seg_dir):
        print("Processing file", file)

        # Extract text from image using default parameters
        path = os.path.join(seg_dir, file)
        words = extract_text(path)

        was_padded = False
        
        # If text was not found, then try harder. Our hypothesis is that some
        # of these images might be words right up against the border.
        PADDING = 10
        if not words:
            print("Nothing found on first pass, might be a self-contained line of text. " +
                  "Trying more aggressive extraction...")

            # Add a little padding to the image on the edges (seems to be necessary
            # for detecting some of the characters when the word is right
            # up against the edges of the image).
            if not os.path.exists("padded"):
                os.makedirs("padded")
            padded_path = os.path.join("padded", file)
            padded = cv2.copyMakeBorder(
                    cv2.imread(path),
                    PADDING,
                    PADDING,
                    PADDING,
                    PADDING,
                    cv2.BORDER_CONSTANT,
                    value=[255, 255, 255] # add white border
                    )
            cv2.imwrite(padded_path, padded)
            path = padded_path
            was_padded = True

            # Redo extraction on the padded image.
            words = extract_text(padded_path, config=r'')
        
        if not words:
            print("I didn't find any text in this image.")
            print()
            continue

        # Check to see if the text is right up against the edges of the image. 
        image_width, image_height = Image.open(path).size
        MARGIN = 2  # set arbitrarily, based on tests
        left_margin = MARGIN
        right_margin = image_width - MARGIN
        top_margin = MARGIN
        bottom_margin = image_height - MARGIN

        if was_padded:
            left_margin += PADDING
            right_margin -= PADDING
            top_margin += PADDING
            bottom_margin -= PADDING

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
                if VERBOSE:
                    print("Left hit", word)
                on_left_border = True
            if word.r > right_margin:
                if VERBOSE:
                    print("Right hit", word)
                on_right_border = True
            if word.t < top_margin and word.text not in POSSIBLE_HORIZONTAL_LINE_CHARS:
                if VERBOSE:
                    print("Top hit", word)
                on_top_border = True
            if word.b > bottom_margin and word.text not in POSSIBLE_HORIZONTAL_LINE_CHARS:
                if VERBOSE:
                    print("Bottom hit", word)
                on_bottom_border = True

        if on_left_border and on_right_border and on_top_border and on_bottom_border:
            print("I found text in this image: '", word.text, "'")
            print("It looks like it's right up against the edges, so it's probably just a word " +
                  "or just a line of text and nothing else. Filter out this segment.")
        else:
            print("I found text in the image, but it's in the middle, probably not just a word, but "
                  "something more.")
        print()

