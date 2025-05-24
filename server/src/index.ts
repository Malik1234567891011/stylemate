import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'

dotenv.config()
const app = express()
const PORT = process.env.PORT || 5001;

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

// Error handler
app.use((err: any, req: any, res: any, next: any) => {
  console.error('🔥 Error:', err.message)
  res.status(500).send({ error: err.message })
})

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`)
})
