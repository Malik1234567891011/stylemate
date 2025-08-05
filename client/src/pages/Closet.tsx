import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Tippy from '@tippyjs/react';
import 'tippy.js/dist/tippy.css';
import '../styles/Closet.css';

interface ClosetItem {
  vector: number[];
  timestamp: number;
  filename: string;
  type?: string;
}

interface SuggestionItem {
  image_url: string;
  title: string;
  url: string;
  score: number;
}

const Closet: React.FC = () => {
  const [closet, setCloset] = useState<ClosetItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<SuggestionItem[]>([]);
  const [suggestionError, setSuggestionError] = useState<string | null>(null);

  const [mode, setMode] = useState<'aesthetic' | 'styleDNA' | 'single'>('aesthetic');
  const [selectedTimestamp, setSelectedTimestamp] = useState<number | null>(null);
  const [minPrice, setMinPrice] = useState<number | null>(null);
  const [maxPrice, setMaxPrice] = useState<number | null>(null);

  const fetchCloset = () => {
    fetch('http://127.0.0.1:8000/closet')
      .then((res) => res.json())
      .then((data) => {
        setCloset(data.reverse());
        setLoading(false);
      })
      .catch((err) => {
        console.error('❌ Failed to fetch closet:', err);
        setLoading(false);
      });
  };

  useEffect(() => {
    setMounted(true);
    fetchCloset();
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadError(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("saveImage", "true");

    axios
      .post("http://127.0.0.1:8000/upload_fit", formData)
      .then(() => {
        setUploading(false);
        fetchCloset();
      })
      .catch((err) => {
        console.error("❌ Upload error:", err);
        setUploadError("Upload failed. Try again.");
        setUploading(false);
      });
  };

  const fetchSuggestions = () => {
    setSuggestionError(null);

    const params: Record<string, string> = {
      mode: mode === 'styleDNA' ? 'expand' : mode === 'single' ? 'single' : 'aesthetic',
      k: '5',
    };

    if (mode === 'single' && selectedTimestamp) {
      params.timestamp = selectedTimestamp.toString();
    }

    if (minPrice !== null) params.min_price = minPrice.toString();
    if (maxPrice !== null) params.max_price = maxPrice.toString();

    const queryString = new URLSearchParams(params).toString();
    fetch(`http://127.0.0.1:8000/recommend_mode?${queryString}`)
      .then((res) => {
        if (!res.ok) throw new Error("Suggestion fetch failed");
        return res.json();
      })
      .then((data) => setSuggestions(data))
      .catch((err) => {
        console.error("❌ Suggestion fetch error:", err);
        setSuggestionError("Could not fetch suggestions.");
      });
  };

  const handleDelete = (index: number) => {
    const updated = [...closet];
    updated.splice(index, 1);
    setCloset(updated);

    fetch("http://127.0.0.1:8000/closet", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updated),
    }).catch((err) => {
      console.error("❌ Delete failed:", err);
    });
  };

  return (
    <div className={`closet-container ${mounted ? 'fade-in' : ''}`}>
      <div className="closet-card-container">
        <h2 className="closet-title">🧳 Your Closet</h2>

        {/* Upload Button */}
        <label className="upload-btn" style={{ marginBottom: "1rem" }}>
          Upload New Fit
          <input
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            style={{ display: "none" }}
          />
        </label>
        {uploading && <p className="loading-text">Uploading...</p>}
        {uploadError && <p className="error-text">{uploadError}</p>}

        {/* Match Mode Selector */}
        {closet.length > 0 && (
          <>
            <div style={{ marginBottom: "1rem" }}>
              <label>
                Match Mode:
                <Tippy
                  content={
                    <div style={{ maxWidth: '300px', fontSize: '0.9rem' }}>
                      <strong>🎯 Closest Match to Specific Outfit</strong>: Finds items that match one selected outfit.<br /><br />
                      <strong>🧬 Ultra Similar to Style DNA</strong>: Finds items that match your entire wardrobe's style average.<br /><br />
                      <strong>🌈 Aesthetic Match</strong>: Loosely matches the vibe of your wardrobe for variety.
                    </div>
                  }
                  placement="right"
                  animation="shift-away"
                >
                  <span style={{ cursor: 'pointer', marginLeft: '0.5rem' }}>❓</span>
                </Tippy>
              </label>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as any)}
                style={{ marginLeft: "1rem", marginRight: "1rem" }}
              >
                <option value="aesthetic">Aesthetic Match</option>
                <option value="styleDNA">Ultra Similar to Style DNA</option>
                <option value="single">Closest Match to Specific Outfit</option>
              </select>

              {mode === "single" && (
                <select
                  onChange={(e) => setSelectedTimestamp(Number(e.target.value))}
                  defaultValue=""
                >
                  <option value="" disabled>Select Outfit</option>
                  {closet.map((item, i) => (
                    <option key={i} value={item.timestamp}>
                      Outfit #{i + 1} ({new Date(item.timestamp * 1000).toLocaleDateString()})
                    </option>
                  ))}
                </select>
              )}
            </div>

            {/* Price Filter */}
            <div style={{ marginBottom: "1rem" }}>
              <label>Min Price: </label>
              <input type="number" onChange={(e) => setMinPrice(parseFloat(e.target.value))} />
              <label style={{ marginLeft: "1rem" }}>Max Price: </label>
              <input type="number" onChange={(e) => setMaxPrice(parseFloat(e.target.value))} />
            </div>

            {/* Suggestion Button */}
            <button className="upload-btn" style={{ marginBottom: "2rem" }} onClick={fetchSuggestions}>
              Get AI Suggestions
            </button>
          </>
        )}

        {/* Closet Items */}
        {loading ? (
          <p className="loading-text">Loading your saved outfits...</p>
        ) : closet.length === 0 ? (
          <div className="empty-closet-card">
            <p className="empty-text">Your closet’s looking empty right now.</p>
          </div>
        ) : (
          <div className="closet-grid">
            {closet.map((item, i) => (
              <div key={i} className="closet-card">
                <div className="closet-img-placeholder">
                  {item.filename ? (
                    <img
                      src={`http://127.0.0.1:8000/uploads/${item.filename}`}
                      alt={`Outfit ${i + 1}`}
                      className="closet-img"
                    />
                  ) : (
                    <span>🖼️</span>
                  )}
                </div>
                <div className="closet-info">
                  <p className="closet-filename">Uploaded Outfit #{i + 1}</p>
                  <p className="closet-timestamp">
                    {new Date(item.timestamp * 1000).toLocaleString()}
                  </p>
                  <button
                    className="delete-btn"
                    onClick={() => handleDelete(i)}
                  >
                    ❌ Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Suggestions */}
        {suggestions.length > 0 && (
  <>
    <h3 className="suggestion-title">🧠 Suggested Items</h3>
    <div className="closet-grid">
      {suggestions.map((item, i) => (
        <a
          key={i}
          href={item.url}
          target="_blank"
          rel="noopener noreferrer"
          className="closet-card"
        >
          <div className="closet-img-placeholder">
            {/* ⛔️ Removed image entirely */}
            {/* <img src={item.image_url} alt={item.title} className="closet-img" /> */}
            <div className="no-image-box">🧥</div>
          </div>
          <div className="closet-info">
            <p className="closet-filename">{item.title}</p>
            {/* ⛔️ Removed score display */}
          </div>
        </a>
      ))}
    </div>
  </>
)}

        {suggestionError && <p className="error-text">{suggestionError}</p>}
      </div>
    </div>
  );
};

export default Closet;
