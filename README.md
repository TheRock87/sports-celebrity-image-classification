# 🏆 Sports Celebrity Classifier

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B" alt="Streamlit 1.32.0">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status: Active">
</div>

A modern, user-friendly web application that identifies sports celebrities from uploaded images using a pre-trained machine learning model. Built with Streamlit, this application provides an intuitive interface for users to upload images and receive instant predictions with confidence scores.

## ✨ Key Features

- **Modern UI/UX**: Clean, responsive design that works on both desktop and mobile devices
- **Multiple Input Methods**: Upload your own images or try with built-in samples
- **Detailed Predictions**: View top 3 predictions with confidence levels and visual indicators
- **Image Analysis**: See technical details about uploaded images (dimensions, format, etc.)
- **Performance Optimized**: Efficient image processing and model inference
- **Error Handling**: Comprehensive error messages and user guidance

## 🎯 Supported Athletes

The model can identify the following sports celebrities:

- Roger Federer (Tennis)
- Morgan Gibbs-White (Soccer)
- Serena Williams (Tennis)
- Lionel Messi (Soccer)
- Maria Sharapova (Tennis)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for first-time setup)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sports-celebrity-classifier.git
   cd sports-celebrity-classifier
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Streamlit server**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your web browser and navigate to `http://localhost:8501`
   - The application should automatically open in your default browser

3. **Using the application**
   - Click "Browse files" to upload an image of a sports celebrity
   - Or select a sample image from the dropdown menu
   - View the prediction results and confidence scores

## 🖼️ Sample Images

Try these sample images to test the application:

1. [Roger Federer Sample](samples/roger_federer.jpg)
2. [Lionel Messi Sample](samples/lionel_messi.jpg)
3. [Serena Williams Sample](samples/serena_williams.jpg)

## 🛠️ Technical Details

### Model Architecture

The application uses a pre-trained machine learning model with the following specifications:

- **Input**: 100x100 grayscale images (10,000 features)
- **Output**: Probabilities for 5 different sports celebrities
- **Model Type**: Pre-trained classifier (saved as `model/best_model.pkl`)
- **Accuracy**: 81.25% (0.8125) on the test dataset

### Project Structure

```
sports-celebrity-classifier/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── model/
│   └── best_model.pkl    # Pre-trained model weights
├── class_dictionary.json   # Class label mappings
└── samples/               # Sample images for testing
    ├── roger_federer.jpg
    ├── lionel_messi.jpg
    └── serena_williams.jpg
```

### Dependencies

Key dependencies include:

- `streamlit==1.32.0`: Web application framework
- `scikit-learn==1.4.0`: Machine learning library
- `opencv-python==4.9.0.80`: Image processing
- `Pillow==10.2.0`: Image handling
- `numpy==1.24.3`: Numerical computations
- `joblib==1.3.2`: Model serialization

## 📱 Mobile Experience

The application features a responsive design that adapts to different screen sizes:

- Optimized for touch interactions
- Adaptive layout for various screen dimensions
- Fast loading times on mobile networks
- Touch-friendly form elements and buttons

## 🎨 UI Components

1. **Image Upload Area**
   - Drag-and-drop interface
   - File browser integration
   - Support for JPG, JPEG, and PNG formats

2. **Prediction Results**
   - Top prediction with confidence score
   - Visual confidence bars
   - Detailed breakdown of top predictions

3. **Image Information**
   - Preview of uploaded image
   - Technical specifications (dimensions, format)
   - Loading indicators during processing

## 📝 Usage Tips

For best results:

1. Use clear, well-lit photos
2. Front-facing portraits work best
3. Ensure the face is clearly visible
4. Avoid group photos or images with multiple faces
5. Recommended minimum resolution: 200x200 pixels

## 🔧 Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure the `model` directory exists and contains `best_model.pkl`
   - Verify file permissions

2. **Dependency conflicts**
   - Use a virtual environment
   - Ensure you're using the exact package versions from `requirements.txt`

3. **Image upload issues**
   - Check file format (supports .jpg, .jpeg, .png)
   - Verify file size (max 200MB)
   - Ensure the image is not corrupted

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The machine learning model was trained using a custom dataset
- Built with ❤️ using Streamlit
- Inspired by computer vision and sports analytics applications

## 📞 Contact

For questions or feedback, please open an issue on GitHub or contact the maintainers.

## 📝 Note

For best results, use clear, front-facing images of the sports celebrities. The model works best with well-lit, high-quality images where the face is clearly visible.
