import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'
import multer from 'multer'

dotenv.config()
const app = express()
const PORT = process.env.PORT || 5001

// Setup Multer to parse incoming images
const upload = multer({ storage: multer.memoryStorage() })

// CORS debug
console.log('🛡️ Setting up CORS middleware...')
app.use(cors({
  origin: (origin, callback) => {
    console.log(`🌐 Incoming request from origin: ${origin}`)
    if (!origin || origin === 'http://localhost:5173') {
      callback(null, true)
    } else {
      callback(new Error('❌ Not allowed by CORS'))
    }
  }
}))

app.use(express.json())

// Debug incoming request logging middleware
app.use((req, res, next) => {
  console.log(`➡️ ${req.method} ${req.url}`)
  next()
})

// Root route
app.get('/', (req, res) => {
  console.log('✅ GET / hit — responding with API status')
  res.send('StyleMate API Running')
})

// 👕 Mock style analysis endpoint
app.post('/api/analyze-style', upload.array('images'), (req, res) => {
  console.log('📸 Received images:', req.files?.length)

  // Simulated analysis results
  const mockStyles = [
    {
      style: 'Vintage',
      tags: ['earth tones', 'layered', 'retro cuts'],
      example_items: [
        {
          brand: "Levi's",
          item_name: 'Corduroy Jacket',
          image_url: 'https://example.com/jacket.jpg',
          price_range: '$$'
        }
      ]
    },
    {
      style: 'Streetwear',
      tags: ['baggy', 'graphic tees', 'sneakers'],
      example_items: [
        {
          brand: 'Supreme',
          item_name: 'Box Logo Tee',
          image_url: 'https://example.com/supreme.jpg',
          price_range: '$$$'
        }
      ]
    },
    {
      style: 'Minimalist',
      tags: ['neutral colors', 'clean lines', 'monochrome'],
      example_items: [
        {
          brand: 'COS',
          item_name: 'Wool Blend Coat',
          image_url: 'https://example.com/cos-coat.jpg',
          price_range: '$$'
        }
      ]
    }
  ]

  const chosen = mockStyles[Math.floor(Math.random() * mockStyles.length)]
  res.send(chosen)
})

// Error handler
app.use((err: any, req: any, res: any, next: any) => {
  console.error('🔥 Error:', err.message)
  res.status(500).send({ error: err.message })
})

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`)
})
