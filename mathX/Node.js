const express = require('express');
const fileUpload = require('express-fileupload');
const mathjs = require('mathjs');
const { createWorker } = require('tesseract.js');
const fetch = require('node-fetch');

const app = express();
app.use(fileUpload());
app.use(express.json()); // Middleware to parse JSON requests

const WOLFRAM_API_KEY = 'YOUR_WOLFRAM_ALPHA_API_KEY'; // Replace with your actual API key
const worker = createWorker();

// Helper function to query Wolfram Alpha
async function queryWolframAlpha(problem) {
  const wolframUrl = `http://api.wolframalpha.com/v2/query?input=${encodeURIComponent(problem)}&format=plaintext&output=JSON&appid=${WOLFRAM_API_KEY}`;
  const response = await fetch(wolframUrl);
  const data = await response.json();
  // Parse the response to extract the solution
  return data.queryresult.pods[0].subpods[0].plaintext;
}

// Endpoint for solving math problems from text
app.post('/solve-text', async (req, res) => {
  const problem = req.body.problem;
  try {
    let solution;
    try {
      solution = mathjs.evaluate(problem);
    } catch {
      // Fallback to Wolfram Alpha for complex problems
      solution = await queryWolframAlpha(problem);
    }
    res.json({ problem, solution });
  } catch (error) {
    console.error('Error solving problem:', error);
    res.status(500).send('Error solving math problem.');
  }
});

// Endpoint for solving math problems from uploaded images
app.post('/solve-image', async (req, res) => {
  if (!req.files || !req.files.image) {
    return res.status(400).send('No image was uploaded.');
  }

  let imageFile = req.files.image;
  try {
    await worker.load();
    await worker.loadLanguage('eng');
    await worker.initialize('eng');
    const { data: { text } } = await worker.recognize(imageFile.data);

    let solution;
    try {
      solution = mathjs.evaluate(text);
    } catch {
      // Fallback to Wolfram Alpha for complex problems
      solution = await queryWolframAlpha(text);
    }
    
    await worker.terminate();
    res.json({ problem: text, solution });
  } catch (error) {
    console.error('Error processing image:', error);
    res.status(500).send('Error solving math problem from image.');
  }
});

// Additional functionality as needed...

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`mathX server running on port ${PORT}`);
});
