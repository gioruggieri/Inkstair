import React, { useState } from "react";
import axios from "axios";

const Badge = ({ value }) => {
  const color = value === true ? "bg-green-600" : value === false ? "bg-red-600" : "bg-gray-500";
  return <span className={`text-white px-3 py-1 rounded-full text-xs font-semibold ${color}`}>{String(value)}</span>;
};

const ScoreBar = ({ label, score }) => {
  const width = `${Math.min(Math.max(score * 100, 0), 100)}%`;
  const color = score >= 0.7 ? "bg-green-500" : score >= 0.4 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="mb-5">
      <div className="flex justify-between text-sm mb-1 font-medium">
        <span>{label}</span>
        <span>{score}</span>
      </div>
      <div className="w-full h-3 bg-gray-300 rounded-full overflow-hidden">
        <div className={`h-full ${color}`} style={{ width }}></div>
      </div>
    </div>
  );
};

export default function ManuscriptAnalyzer() {
  const [file, setFile] = useState(null);
  const [genres, setGenres] = useState("giallo, rosa, fantasy, psicologico");
  const [keywords, setKeywords] = useState("emozione, sincerità, altruismo, generosità");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("accepted_genres", genres);
    formData.append("trend_keywords", keywords);

    try {
      setLoading(true);
      const response = await axios.post("http://localhost:8005/analyze", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
    } catch (err) {
      console.error("Errore durante l'invio:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-100 via-white to-indigo-200 py-10 px-4">
      <div className="max-w-3xl mx-auto bg-white p-10 rounded-3xl shadow-xl border border-gray-300">
        <h1 className="text-4xl font-extrabold text-indigo-800 mb-8 text-center flex items-center justify-center gap-3">
          <span role="img">📘</span> InkStair Manuscript Analyzer
        </h1>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block font-semibold text-gray-700 mb-2">📄 Carica il tuo manoscritto (PDF):</label>
            <input
              type="file"
              accept=".pdf"
              onChange={(e) => setFile(e.target.files[0])}
              className="block w-full border border-gray-300 px-4 py-2 rounded-lg shadow-sm text-sm"
              required
            />
          </div>

          <div>
            <label className="block font-semibold text-gray-700 mb-2">🎭 Generi accettati:</label>
            <input
              type="text"
              value={genres}
              onChange={(e) => setGenres(e.target.value)}
              className="w-full border border-gray-300 px-4 py-2 rounded-lg text-sm"
            />
          </div>

          <div>
            <label className="block font-semibold text-gray-700 mb-2">📈 Parole chiave di trend:</label>
            <input
              type="text"
              value={keywords}
              onChange={(e) => setKeywords(e.target.value)}
              className="w-full border border-gray-300 px-4 py-2 rounded-lg text-sm"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white text-lg font-semibold py-2 rounded-lg shadow transition duration-200"
          >
            {loading ? "🔍 Analisi in corso..." : "🚀 Avvia analisi"}
          </button>
        </form>

        {result && (
          <div className="mt-12 bg-white p-6 rounded-xl shadow-md border border-gray-200">
            <h2 className="text-2xl font-bold text-indigo-800 mb-4 text-center">📊 Risultati dell'Analisi</h2>
            <ul className="space-y-2 text-gray-700 text-sm">
              <li><strong>✔️ Correttezza grammaticale:</strong> {result.grammar_score} / 100</li>
              <li><strong>📘 Genere rilevato:</strong> <span className="italic">{result.genre}</span></li>
              <li><strong>✅ Accettato per genere:</strong> <Badge value={result.is_accepted_by_genre} /></li>
              <li><strong>💬 Sentiment:</strong> {result.sentiment}</li>
              <li><strong>🎯 Match target emotivo:</strong> <Badge value={result.sentiment_match} /></li>
            </ul>

            <div className="mt-6">
              <ScoreBar label="Sentiment Score" score={result.sentiment_score || 0} />
              <ScoreBar label="Market Score" score={result.market_score || 0} />
            </div>

            <div className="mt-4">
              <strong>🔥 In linea con i trend di mercato:</strong> <Badge value={result.market_match} />
            </div>

            <div className="mt-4 text-sm">
              <strong>🎯 Generi accettati:</strong> <code>{result.accepted_genres?.join(", ")}</code><br />
              <strong>🔍 Trend analizzati:</strong> <code>{result.trend_keywords?.join(", ")}</code>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
