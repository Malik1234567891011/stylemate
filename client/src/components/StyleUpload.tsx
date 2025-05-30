// client/src/StyleUpload.tsx

import { useState } from 'react'
import axios from 'axios'

interface Product {
  title: string
  price: string
  url: string
  score: number
}

const StyleUpload = () => {
  const [file, setFile] = useState<File | null>(null)
  const [results, setResults] = useState<Product[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setResults(null)
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select an image.')
      return
    }
    setLoading(true)
    setError(null)
    setResults(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await axios.post<Product[]>(
        'http://127.0.0.1:8000/recommend?k=5',
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      )
      setResults(res.data)
    } catch (err: any) {
      console.error(err)
      setError('Error fetching recommendations.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: '2rem' }}>
      <h2>Upload an Outfit Photo</h2>
      <input type="file" accept="image/*" onChange={handleFileChange} />

      {file && (
        <button
          onClick={handleUpload}
          disabled={loading}
          style={{ marginTop: '1rem' }}
        >
          {loading ? 'Analyzing...' : 'Get Recommendations'}
        </button>
      )}

      {error && (
        <div style={{ marginTop: '1rem', color: 'red' }}>{error}</div>
      )}

      {results && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(200px,1fr))',
            gap: '1rem',
            marginTop: '2rem',
          }}
        >
          {results.map((prod) => (
            <div
              key={prod.url}
              style={{
                border: '1px solid #ccc',
                padding: '1rem',
                borderRadius: '8px',
              }}
            >
              <a href={prod.url} target="_blank" rel="noopener noreferrer">
                <h3>{prod.title}</h3>
                <p>{prod.price}</p>
                <p>Score: {prod.score.toFixed(3)}</p>
              </a>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default StyleUpload
