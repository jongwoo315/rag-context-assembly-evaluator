import { useEffect, useState } from 'react'
import './App.css'

const API_BASE = '/api'

function App() {
  const [corpusId, setCorpusId] = useState('')
  const [questionsId, setQuestionsId] = useState('')
  const [config, setConfig] = useState({ k: 3, token_budget: 512, rerank_cutoff: 0.3, mmr_lambda: 0.5 })
  const [run, setRun] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`${API_BASE}/config/defaults`)
      .then((r) => r.json())
      .then((data) => {
        setCorpusId(data.corpus_id)
        setQuestionsId(data.questions_id)
      })
      .catch((e) => setError(e.message))
  }, [])

  const runEval = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/eval/${corpusId}/${questionsId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      const data = await res.json()
      if (data.status !== 'ok') throw new Error(`status ${data.status}`)
      setRun(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const scoreStyle = (value) => ({
    color: value >= 0.7 ? '#16a34a' : value >= 0.4 ? '#ca8a04' : '#dc2626',
    fontWeight: 700,
  })

  return (
    <main className="container">
      <h1>RAG Context Assembly Evaluator</h1>
      <p className="subtitle">
        RAG context-assembly 전략을 golden QA 셋과 비교해 평가하고, 실패 유형별 권고를 생성합니다.
      </p>

      <section className="controls">
        <div className="field">
          <label>Corpus ID</label>
          <input value={corpusId} onChange={(e) => setCorpusId(e.target.value)} />
        </div>
        <div className="field">
          <label>Questions ID</label>
          <input value={questionsId} onChange={(e) => setQuestionsId(e.target.value)} />
        </div>
        <div className="field">
          <label>top-k</label>
          <input
            type="number"
            value={config.k}
            onChange={(e) => setConfig({ ...config, k: Number(e.target.value) })}
          />
        </div>
        <div className="field">
          <label>token budget</label>
          <input
            type="number"
            value={config.token_budget}
            onChange={(e) => setConfig({ ...config, token_budget: Number(e.target.value) })}
          />
        </div>
        <div className="field">
          <label>rerank cutoff</label>
          <input
            type="number"
            step="0.05"
            value={config.rerank_cutoff}
            onChange={(e) => setConfig({ ...config, rerank_cutoff: Number(e.target.value) })}
          />
        </div>
        <div className="field">
          <label>mmr lambda</label>
          <input
            type="number"
            step="0.05"
            value={config.mmr_lambda}
            onChange={(e) => setConfig({ ...config, mmr_lambda: Number(e.target.value) })}
          />
        </div>
        <button className="primary" onClick={runEval} disabled={loading || !corpusId || !questionsId}>
          {loading ? '평가 중…' : '전략 평가 실행'}
        </button>
      </section>

      {error && <div className="error">오류: {error}</div>}

      {run && (
        <section className="results">
          <div className="best">
            최고 전략: <strong>{run.best_strategy}</strong> (F1 {run.best_f1.toFixed(3)})
          </div>
          <table className="result-table">
            <thead>
              <tr>
                <th>전략</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>Context Precision</th>
                <th>청크 수</th>
                <th>토큰 수</th>
                <th>지연(ms)</th>
              </tr>
            </thead>
            <tbody>
              {run.results.map((r) => (
                <tr key={r.strategy} className={r.strategy === run.best_strategy ? 'best-row' : ''}>
                  <td>{r.strategy}</td>
                  <td style={scoreStyle(r.precision)}>{r.precision.toFixed(3)}</td>
                  <td style={scoreStyle(r.recall)}>{r.recall.toFixed(3)}</td>
                  <td style={scoreStyle(r.f1)}>{r.f1.toFixed(3)}</td>
                  <td style={scoreStyle(r.context_precision)}>{r.context_precision.toFixed(3)}</td>
                  <td>{r.chunk_count}</td>
                  <td>{r.token_count}</td>
                  <td>{r.latency_ms}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h2>Failure-mode 분포</h2>
          <div className="failure-grid">
            {run.results.map((r) => (
              <div key={r.strategy} className="failure-card">
                <h3>{r.strategy}</h3>
                <ul>
                  {Object.entries(r.failure_modes).map(([mode, count]) => (
                    <li key={mode}>
                      <span className="mode">{mode}</span>
                      <span className="count">{count}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          <h2>전략별 권고</h2>
          <div className="recommendations">
            {run.results.map((r) =>
              r.recommendations.map((rec, idx) => (
                <div key={`${r.strategy}-${idx}`} className="rec">
                  <span className="rec-strategy">{r.strategy}</span>
                  <span className="rec-text">{rec}</span>
                </div>
              ))
            )}
          </div>
        </section>
      )}
    </main>
  )
}

export default App
