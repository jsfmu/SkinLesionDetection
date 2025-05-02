# Skin Lesion Detection System

This project implements a deep learning-based system for detecting and classifying skin lesions as either malignant or benign. The system consists of a machine learning model trained on the HAM10000 dataset and a web application for user interaction.

## Quick Start

1. Open your web browser and navigate to:
   - Local development: `http://localhost:3000`
   - Production: `https://your-deployed-app-url.com`

2. You'll see the main interface with:
   - Upload button for skin lesion images
   - History of previous analyses
   - Settings menu

3. To analyze a skin lesion:
   - Click "Upload Image" or drag and drop an image
   - Wait for the analysis (usually 2-3 seconds)
   - View the results showing:
     - Classification (Malignant/Benign)
     - Confidence score
     - Visual explanation

## Project Structure

```
skinlesion/
├── HAM10000_images/           # Dataset directory
├── skin_lesion_app/           # Frontend application
├── skin_lesion_detection.py   # Training and model code
├── convert_to_tflite.py      # Model conversion script
├── analyze_data.py           # Data analysis utilities
└── README.md                 # This file
```

## Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- OpenCV
- Flask
- React.js (for frontend)
- Node.js and npm (for frontend)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd skinlesion
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd skin_lesion_app
npm install
```

## Dataset

The project uses the HAM10000 dataset, which contains 10,000 dermatoscopic images of common pigmented skin lesions. The dataset includes 7 different types of lesions:

- Melanoma (mel)
- Melanocytic nevus (nv)
- Basal cell carcinoma (bcc)
- Actinic keratosis / Bowen's disease (akiec)
- Benign keratosis (bkl)
- Dermatofibroma (df)
- Vascular lesion (vasc)

## Training the Model

The model is trained using a CNN architecture with K-fold cross-validation. To train the model:

1. Ensure the HAM10000 dataset is in the correct directory:
```
HAM10000_images/
└── archive/
    ├── HAM10000_images_part_1/
    └── HAM10000_metadata.csv
```

2. Run the training script:
```bash
python skin_lesion_detection.py
```

The training process includes:
- 5-fold cross-validation
- Data augmentation
- Early stopping
- Learning rate reduction
- Model checkpointing

The script will:
- Load the existing model if available
- Train for 100 epochs per fold
- Save the best model from each fold
- Save the overall best model as `skin_lesion_model.h5`

## Converting to TFLite

To convert the trained model to TFLite format for mobile deployment:

```bash
python convert_to_tflite.py
```

This will create a `skin_lesion_model.tflite` file.

## Deployment

### Backend Deployment (Flask)

1. Prepare the backend for production:
```bash
# Install production dependencies
pip install gunicorn

# Create a requirements.txt file
pip freeze > requirements.txt
```

2. Deploy to a cloud platform (e.g., Heroku, AWS, or Google Cloud):

#### Heroku Deployment
```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create a new Heroku app
heroku create your-app-name

# Add Python buildpack
heroku buildpacks:set heroku/python

# Set environment variables
heroku config:set FLASK_ENV=production
heroku config:set FLASK_APP=app.py

# Deploy
git push heroku main
```

#### AWS Elastic Beanstalk Deployment
1. Install AWS CLI and EB CLI
2. Initialize EB application:
```bash
eb init -p python-3.8 your-app-name
eb create your-environment-name
```

3. Deploy:
```bash
eb deploy
```

### Frontend Deployment (React)

1. Build the React application:
```bash
cd skin_lesion_app
npm run build
```

2. Deploy to a hosting service:

#### Netlify Deployment
1. Install Netlify CLI:
```bash
npm install -g netlify-cli
```

2. Deploy:
```bash
netlify deploy
```

#### Vercel Deployment
1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Deploy:
```bash
vercel
```

### Environment Configuration

1. Create a `.env` file in the backend directory:
```
FLASK_ENV=production
FLASK_APP=app.py
DATABASE_URL=your_database_url
```

2. Create a `.env` file in the frontend directory:
```
REACT_APP_API_URL=your_backend_url
```

### Production Considerations

1. Security:
   - Enable HTTPS
   - Set up CORS properly
   - Implement rate limiting
   - Use environment variables for sensitive data

2. Performance:
   - Enable caching
   - Use a CDN for static files
   - Optimize image loading
   - Implement lazy loading

3. Monitoring:
   - Set up error logging
   - Monitor server resources
   - Track API usage
   - Set up alerts for critical issues

## Running the Application Locally

### Backend

1. Start the Flask server:
```bash
python app.py
```

The server will run on `http://localhost:5000`

### Frontend

1. Navigate to the frontend directory:
```bash
cd skin_lesion_app
```

2. Start the development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`

## Usage

1. Open the web application in your browser
2. Upload a skin lesion image
3. The system will analyze the image and provide:
   - Classification (Malignant/Benign)
   - Confidence score
   - Visual explanation of the prediction

## Model Performance

The model achieves the following performance metrics:
- Binary classification (Malignant vs Benign)
- Accuracy: ~85-90% (varies by fold)
- Uses data augmentation to improve generalization
- Implements early stopping to prevent overfitting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HAM10000 dataset creators
- TensorFlow team
- React.js community 