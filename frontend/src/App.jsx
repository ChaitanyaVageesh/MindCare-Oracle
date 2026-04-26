import { useState } from 'react'
import './index.css'

function App() {
  const [formData, setFormData] = useState({
    Age: 30,
    Gender: 'male',
    family_history: 'No',
    benefits: 'Don\'t know',
    care_options: 'Not sure',
    anonymity: 'Don\'t know',
    leave: 'Don\'t know',
    work_interfere: 'Don\'t know'
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({...formData, [e.target.name]: e.target.value});
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      // Assuming API is exposed securely on port 8000
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({...formData, Age: parseInt(formData.Age, 10)})
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed");
      setResult(data);
    } catch (error) {
      alert("Error: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="glass-panel">
        <h1>MindCare Oracle</h1>
        <p className="subtitle">AI-Driven Mental Health Prediction Pipeline</p>
        
        <form onSubmit={handleSubmit}>
          <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem'}}>
            <div className="form-group">
              <label>Age</label>
              <input type="number" name="Age" value={formData.Age} onChange={handleChange} required />
            </div>
            
            <div className="form-group">
              <label>Gender</label>
              <select name="Gender" value={formData.Gender} onChange={handleChange}>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="trans">Trans / Non-binary</option>
              </select>
            </div>

            <div className="form-group">
              <label>Family History</label>
              <select name="family_history" value={formData.family_history} onChange={handleChange}>
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>

            <div className="form-group">
              <label>Benefits Provided?</label>
              <select name="benefits" value={formData.benefits} onChange={handleChange}>
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="Don't know">Don't know</option>
              </select>
            </div>

            <div className="form-group">
              <label>Care Options Known?</label>
              <select name="care_options" value={formData.care_options} onChange={handleChange}>
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="Not sure">Not sure</option>
              </select>
            </div>

            <div className="form-group">
              <label>Anonymity Protected?</label>
              <select name="anonymity" value={formData.anonymity} onChange={handleChange}>
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="Don't know">Don't know</option>
              </select>
            </div>

            <div className="form-group">
              <label>Medical Leave Difficulty</label>
              <select name="leave" value={formData.leave} onChange={handleChange}>
                <option value="Very easy">Very easy</option>
                <option value="Somewhat easy">Somewhat easy</option>
                <option value="Don't know">Don't know</option>
                <option value="Somewhat difficult">Somewhat difficult</option>
                <option value="Very difficult">Very difficult</option>
              </select>
            </div>

            <div className="form-group">
              <label>Work Interfere</label>
              <select name="work_interfere" value={formData.work_interfere} onChange={handleChange}>
                <option value="Never">Never</option>
                <option value="Rarely">Rarely</option>
                <option value="Sometimes">Sometimes</option>
                <option value="Often">Often</option>
                <option value="Don't know">Don't know</option>
              </select>
            </div>
          </div>

          <button type="submit" disabled={loading}>
            {loading ? 'Analyzing...' : 'Predict Outcome'}
          </button>
        </form>

        {result && (
          <div className={`result ${result.treatment ? 'positive' : 'negative'}`}>
            <h2 style={{margin: '0 0 0.5rem 0'}}>
              {result.treatment ? 'Higher Probability of Treatment' : 'Lower Probability of Treatment'}
            </h2>
            <p style={{margin: 0, opacity: 0.8}}>Risk Confidence: {(result.probability * 100).toFixed(1)}%</p>
          </div>
        )}
      </div>

      <div className="glass-panel" style={{alignSelf: 'start'}}>
        <h2>MLOps Pipeline Console</h2>
        <p className="subtitle">Real-time monitoring and experiment tracking access points.</p>
        
        <div className="dashboard-links">
          <a href="http://localhost:5000" target="_blank" rel="noreferrer" className="dash-link">
            <span>MLflow Experiment Tracking <br/><small style={{opacity: 0.6}}>View model parameters & evaluation metrics</small></span>
            <span className="dash-icon">🧪</span>
          </a>
          
          <a href="http://localhost:3001/d/mindcare/mindcare-performance" target="_blank" rel="noreferrer" className="dash-link">
            <span>Grafana Metrics Dashboard <br/><small style={{opacity: 0.6}}>Live throughput, inference speed & error rates</small></span>
            <span className="dash-icon">📊</span>
          </a>

          <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="dash-link">
            <span>FastAPI Swagger UI <br/><small style={{opacity: 0.6}}>Endpoint interface specifications & I/O testing</small></span>
            <span className="dash-icon">⚡</span>
          </a>

          <a href="http://localhost:9090" target="_blank" rel="noreferrer" className="dash-link">
            <span>Prometheus Target Scraper <br/><small style={{opacity: 0.6}}>Hardware & Endpoint level instrumentation targets</small></span>
            <span className="dash-icon">⚙️</span>
          </a>
        </div>
      </div>
    </div>
  )
}

export default App
