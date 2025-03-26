# Stegnography-and-Steganalysis-Tool

## Overview
This project focuses on **Steganography and Steganalysis**, detecting hidden information in images using **Machine Learning models** like CNN and Random Forest. We use **LSB (Least Significant Bit) techniques and other methods** for hiding data within images.

## Features
- **Steganography and Steganalysis**: Implements both hiding and detecting data in images.
- **LSB-based Data Hiding**: Uses LSB and other techniques to embed hidden messages.
- **CNN-based Detection**: Uses MobileNetV2 to classify stego and non-stego images.
- **Random Forest Model**: Extracts image features for steganalysis.
- **Image Preprocessing**: Converts images to grayscale, resizes, and extracts features.
- **Binary Data Conversion**: Uses LSB analysis for detecting hidden information.

## Usage
1. **Train CNN Model** (if needed):  
   Run `CNN Steganalysis.ipynb` to train the model.
2. **Train Random Forest Model** (if needed):  
   Run `Random Forest Steganalysis.ipynb` to train the model.
3. **Hide Data Using LSB**:
   ```bash
   python project.py --hide --image input.png --message "Secret Message" --output stego.png
   ```
4. **Test an Image for Steganography**:
   ```bash
   python test.py --image test.png
   ```

## Files
- **`CNN Steganalysis.ipynb`** - CNN-based steganalysis.
- **`Random Forest Steganalysis.ipynb`** - Random Forest-based steganalysis.
- **`project.py`** - Helper functions for binary conversion and LSB hiding.
- **`test.py`** - Tests an image using the trained model.
- **`mobilenetv2_lsb_model.h5`** - Pre-trained CNN model.
- **`rf_steganography_model.pkl`** - Pre-trained Random Forest model.

## Applications
- **Cybersecurity**: Detect and prevent unauthorized data embedding.
- **Digital Forensics**: Identify steganography techniques in digital evidence.
- **Data Privacy**: Securely hide sensitive information within images.
