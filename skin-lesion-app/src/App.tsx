import React, { useState, useRef } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  Button, 
  Paper, 
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Drawer,
  AppBar,
  Toolbar,
  ThemeProvider,
  createTheme,
  CssBaseline
} from '@mui/material';
import { 
  CameraAlt as CameraIcon, 
  PhotoLibrary as GalleryIcon,
  History as HistoryIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon
} from '@mui/icons-material';
import Webcam from 'react-webcam';
import axios from 'axios';

// Create a theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

interface Prediction {
  result: string;
  confidence: number;
  timestamp: Date;
}

interface Recommendation {
  title: string;
  description: string;
  urgency: 'high' | 'medium' | 'low';
}

function App() {
  const [image, setImage] = useState<string | null>(null);
  const [result, setResult] = useState<string>('');
  const [confidence, setConfidence] = useState<number>(0);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [showCamera, setShowCamera] = useState<boolean>(false);
  const [history, setHistory] = useState<Prediction[]>([]);
  const [showHistory, setShowHistory] = useState<boolean>(false);
  const webcamRef = useRef<Webcam>(null);

  const handleCapture = () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      setImage(imageSrc);
      setShowCamera(false);
      if (imageSrc) {
        processImage(imageSrc);
      }
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const imageSrc = reader.result as string;
        setImage(imageSrc);
        processImage(imageSrc);
      };
      reader.readAsDataURL(file);
    }
  };

  const processImage = async (imageData: string) => {
    setIsProcessing(true);
    try {
      // Convert base64 to blob
      const base64Data = imageData.split(',')[1];
      const blob = await fetch(`data:image/jpeg;base64,${base64Data}`).then(res => res.blob());

      // Create form data
      const formData = new FormData();
      formData.append('image', blob, 'image.jpg');

      // Send to backend
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const { prediction, confidence, recommendation } = response.data;
      
      if (prediction && confidence) {
        setResult(prediction);
        setConfidence(parseFloat(confidence));
        
        // Add to history
        const newPrediction: Prediction = {
          result: prediction,
          confidence: parseFloat(confidence),
          timestamp: new Date(),
        };
        setHistory(prev => [newPrediction, ...prev]);
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (error) {
      console.error('Error processing image:', error);
      setResult('Error processing image');
      setConfidence(0);
    }
    setIsProcessing(false);
  };

  const getRecommendations = (result: string, confidence: number): Recommendation[] => {
    const recommendations: Recommendation[] = [];

    if (result === 'Malignant') {
      if (confidence > 0.8) {
        recommendations.push({
          title: 'Urgent Medical Attention Required',
          description: 'High confidence of malignant lesion detected. Please consult a dermatologist immediately.',
          urgency: 'high'
        });
      } else if (confidence > 0.6) {
        recommendations.push({
          title: 'Schedule Dermatologist Appointment',
          description: 'Moderate confidence of malignant lesion detected. Schedule an appointment with a dermatologist within the next week.',
          urgency: 'medium'
        });
      } else {
        recommendations.push({
          title: 'Professional Evaluation Recommended',
          description: 'Low confidence of malignant lesion detected. Consider consulting a dermatologist for professional evaluation.',
          urgency: 'low'
        });
      }
    } else {
      if (confidence > 0.8) {
        recommendations.push({
          title: 'Regular Monitoring',
          description: 'High confidence of benign lesion detected. Continue regular skin checks and monitor for any changes.',
          urgency: 'low'
        });
      } else {
        recommendations.push({
          title: 'Follow-up Recommended',
          description: 'Moderate confidence of benign lesion detected. Consider professional evaluation for peace of mind.',
          urgency: 'medium'
        });
      }
    }

    // Add general recommendations
    recommendations.push({
      title: 'General Skin Care',
      description: 'Use sunscreen daily, avoid excessive sun exposure, and perform regular self-examinations.',
      urgency: 'low'
    });

    return recommendations;
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Skin Lesion Detector
            </Typography>
            <IconButton 
              color="inherit" 
              onClick={() => setShowHistory(true)}
            >
              <HistoryIcon />
            </IconButton>
          </Toolbar>
        </AppBar>

        <Container maxWidth="md" sx={{ mt: 4 }}>
          <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
            {showCamera ? (
              <Box sx={{ position: 'relative' }}>
                <Webcam
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  style={{ width: '100%', maxHeight: '400px' }}
                />
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleCapture}
                  sx={{ mt: 2 }}
                >
                  Capture
                </Button>
              </Box>
            ) : (
              <Box>
                {image ? (
                  <img 
                    src={image} 
                    alt="Captured" 
                    style={{ 
                      width: '100%', 
                      maxHeight: '400px', 
                      objectFit: 'contain' 
                    }} 
                  />
                ) : (
                  <Box 
                    sx={{ 
                      height: '400px', 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      border: '2px dashed #ccc',
                      borderRadius: 1
                    }}
                  >
                    <Typography variant="h6" color="textSecondary">
                      No image selected
                    </Typography>
                  </Box>
                )}
              </Box>
            )}

            <Box sx={{ mt: 2, display: 'flex', gap: 2, justifyContent: 'center' }}>
              <Button
                variant="contained"
                startIcon={<CameraIcon />}
                onClick={() => setShowCamera(true)}
              >
                Take Picture
              </Button>
              <Button
                variant="contained"
                startIcon={<GalleryIcon />}
                component="label"
              >
                Upload Image
                <input
                  type="file"
                  hidden
                  accept="image/*"
                  onChange={handleFileUpload}
                />
              </Button>
            </Box>
          </Paper>

          {isProcessing && (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
              <CircularProgress />
            </Box>
          )}

          {result && (
            <>
              <Paper 
                elevation={3} 
                sx={{ 
                  p: 3, 
                  mb: 3,
                  bgcolor: result === 'Malignant' ? 'error.light' : 'success.light',
                  color: result === 'Malignant' ? 'error.contrastText' : 'success.contrastText'
                }}
              >
                <Typography variant="h5" gutterBottom>
                  Result: {result}
                </Typography>
                <Typography variant="h6">
                  Confidence: {(confidence).toFixed(2)}%
                </Typography>
              </Paper>

              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Recommendations
                </Typography>
                {getRecommendations(result, confidence).map((rec, index) => (
                  <Box 
                    key={index} 
                    sx={{ 
                      mb: 2,
                      p: 2,
                      borderRadius: 1,
                      bgcolor: rec.urgency === 'high' ? 'error.light' : 
                              rec.urgency === 'medium' ? 'warning.light' : 
                              'info.light'
                    }}
                  >
                    <Typography variant="subtitle1" fontWeight="bold">
                      {rec.title}
                    </Typography>
                    <Typography variant="body2">
                      {rec.description}
                    </Typography>
                  </Box>
                ))}
              </Paper>
            </>
          )}
        </Container>

        <Drawer
          anchor="right"
          open={showHistory}
          onClose={() => setShowHistory(false)}
        >
          <Box sx={{ width: 350, p: 2 }}>
            <Typography variant="h6" gutterBottom>
              History
            </Typography>
            <List>
              {history.map((item, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    {item.result === 'Malignant' ? (
                      <WarningIcon color="error" />
                    ) : (
                      <CheckCircleIcon color="success" />
                    )}
                  </ListItemIcon>
                  <ListItemText
                    primary={item.result}
                    secondary={`Confidence: ${(item.confidence * 100).toFixed(2)}%\n${item.timestamp.toLocaleString()}`}
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        </Drawer>
      </Box>
    </ThemeProvider>
  );
}

export default App;
