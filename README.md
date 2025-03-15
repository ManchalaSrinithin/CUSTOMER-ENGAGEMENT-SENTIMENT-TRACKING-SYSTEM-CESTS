
# CUSTOMER ENGAGEMENT SENTIMENT TRACKING SYSTEM (CESTS) 

The **Customer Engagement Sentiment Tracking System (CESTS)** is designed to track, analyze, and visualize customer sentiments across various engagement channels. It utilizes sentiment analysis techniques to classify customer feedback and interactions into predefined categories, allowing businesses to better understand customer emotions and improve decision-making processes.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

The CESTS aims to help businesses track the sentiments of customer interactions and engagement data. It leverages sentiment analysis algorithms and provides a user-friendly interface to view and analyze the sentiment trends. The project also includes audio transcription features using OpenAI's Whisper model.

## Features

- **Sentiment Analysis**: Automatically classify customer feedback into categories such as Positive, Negative, Neutral, etc., using an LSTM-based model built with TensorFlow.
- **Audio Transcription**: Transcribe audio feedback into text using the Whisper ASR model.
- **Engagement Tracking**: Monitor and track customer engagement data in real-time.
- **Visualizations**: View sentiment trends over time with graphs and charts.
- **Real-time Updates**: Continuously track customer feedback and engagement.

## Technologies Used

### Backend:
- **Flask**: A lightweight web framework for Python, used to build the API endpoints (`/analyze`, `/transcribe`, `/train`, `/upload`).
- **TensorFlow**: A deep learning framework for building, training, and deploying machine learning models, specifically for sentiment analysis.
- **Whisper**: An open-source automatic speech recognition (ASR) model developed by OpenAI, used to transcribe audio files to text.
- **NLTK (Natural Language Toolkit)**: A Python library for natural language processing, specifically used for sentiment analysis with the VADER sentiment analysis tool.
- **Pickle**: Used to serialize and save the tokenizer object.
- **requests**: A Python HTTP library used to make requests, e.g., sending transcriptions to the `/analyze` endpoint.
- **dotenv**: A Python library used to load environment variables from a `.env` file.

### Frontend:
- **React**: A JavaScript library for building user interfaces, used for creating the frontend of the application.
- **Axios**: A promise-based HTTP client for the browser and Node.js, used to make API requests to the backend.
- **Bootstrap**: A front-end framework for developing responsive and mobile-first websites, used to style the frontend application.
- **React Router**: A routing library for React, used to navigate between different components in the frontend.
- **Flask-CORS**: A Flask extension to handle Cross-Origin Resource Sharing (CORS), enabling the backend API to be accessed from different origins.

### Cloud/Infrastructure:
- **FFmpeg**: A multimedia framework used for audio and video file processing, included in the environment to support Whisper transcription.

### Libraries/Tools:
- **TensorFlow Keras**: Used to create, train, and evaluate a deep learning model (LSTM with embedding layers) for sentiment analysis.
- **Python's logging module**: Used for logging messages for debugging and monitoring purposes.
- **Python's OS and tempfile modules**: Used for handling file management, such as saving temporary files for audio uploads.

## Installation Instructions

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/ManchalaSrinithin/CUSTOMER-ENGAGEMENT-SENTIMENT-TRACKING-SYSTEM-CESTS.git
   ```

2. Navigate to the project directory:

   ```bash
   cd CESTS
   ```

3. Install dependencies for the backend:

   ```bash
   pip install Flask TensorFlow Whisper nltk requests python-dotenv flask-cors
   ```

4. Install **FFmpeg** for audio processing. On most systems, you can install FFmpeg using the following commands:
   - **Ubuntu**:

     ```bash
     sudo apt install ffmpeg
     ```

   - **Mac** (via Homebrew):

     ```bash
     brew install ffmpeg
     ```

5. Install dependencies for the frontend (if using React):

   Navigate to the frontend directory and run:

   ```bash
   npm install
   ```

6. Set up the environment variables:
   - Create a `.env` file and add any necessary environment variables (e.g., API keys or model paths).

7. Run the backend server:

   ```bash
   python app.py
   ```

### Frontend:

1. Navigate to the root directory:
   ```bash
   cd path/to/your
   ```

2. Install the dependencies:
   ```bash
   npm install
   ```

3. Run the frontend server:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:3000`.


## Usage

Once the application is running, you can interact with the system through the following endpoints:

- **POST /analyze**: Analyzes the sentiment of customer feedback (text).
- **POST /transcribe**: Transcribes audio feedback into text using the Whisper ASR model.
- **POST /train**: Trains the sentiment analysis model with new data.
- **POST /upload**: Uploads audio files for transcription and analysis.

## Contributing

We welcome contributions to the CESTS project! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgements

- **TensorFlow**: For building and training the deep learning model for sentiment analysis.
- **Whisper**: For audio transcription.
- **NLTK**: For sentiment analysis with the VADER sentiment analysis tool.
- **FFmpeg**: For processing audio and video files.

