import { useState } from 'react'
import axios from 'axios'

const StyleUpload = () => {
  const [files, setFiles] = useState<File[]>([])
  const [result, setResult] = useState<string>('')

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files))
    }
  }

  const handleUpload = async () => {
    const formData = new FormData()
    files.forEach((file) => formData.append('images', file))

    try {
      const res = await axios.post<{ message: string }>(
        'http://localhost:5001/api/analyze-style',
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      )
      setResult(res.data.message)
    } catch (err) {
      console.error(err)
      setResult('Error analyzing your style.')
    }
  }

  return (
    <div style={{ padding: '2rem' }}>
      <h2>Upload 3–5 Outfit Images</h2>
      <input type="file" accept="image/*" multiple onChange={handleFileChange} />
      <br />
      {files.length > 0 && (
        <button onClick={handleUpload} style={{ marginTop: '1rem' }}>
          Analyze My Style
        </button>
      )}
      {result && (
        <div style={{ marginTop: '1rem' }}>
          <strong>Result:</strong> {result}
        </div>
      )}
    </div>
  )
}

export default StyleUpload
