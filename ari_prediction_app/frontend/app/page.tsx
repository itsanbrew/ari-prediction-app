'use client';

import React, { useState, useEffect } from 'react';

interface Symptoms {
  coryza: boolean;
  wheezing: boolean;
  sore_throat: boolean;
  nasal_congestion: boolean;
  fever: boolean;
  cough: boolean;
  chills: boolean;
  difficulty_breathing: boolean;
  coughing_up_sputum: boolean;
  shortness_of_breath: boolean;
  chest_tightness: boolean;
  vomiting: boolean;
}

interface PatientInfo {
  name: string;
  age: string;
  education: string;
  religion: string;
}

interface PredictionResult {
  prediction: string;
  probability: number;
  confidence: string;
}

export default function Home() {
  const [symptoms, setSymptoms] = useState<Symptoms>({
    coryza: false,
    wheezing: false,
    sore_throat: false,
    nasal_congestion: false,
    fever: false,
    cough: false,
    chills: false,
    difficulty_breathing: false,
    coughing_up_sputum: false,
    shortness_of_breath: false,
    chest_tightness: false,
    vomiting: false,
  });

  const [patientInfo, setPatientInfo] = useState<PatientInfo>({
    name: '',
    age: '',
    education: '',
    religion: '',
  });

  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset all state on component mount (page refresh)
  useEffect(() => {
    // This ensures clean state on page refresh
    setSymptoms({
      coryza: false,
      wheezing: false,
      sore_throat: false,
      nasal_congestion: false,
      fever: false,
      cough: false,
      chills: false,
      difficulty_breathing: false,
      coughing_up_sputum: false,
      shortness_of_breath: false,
      chest_tightness: false,
      vomiting: false,
    });
    setPatientInfo({
      name: '',
      age: '',
      education: '',
      religion: '',
    });
    setResult(null);
    setError(null);
    setLoading(false);
  }, []);

  const handleSymptomChange = (symptom: keyof Symptoms) => {
    setSymptoms(prev => ({
      ...prev,
      [symptom]: !prev[symptom]
    }));
  };

  const handlePatientInfoChange = (field: keyof PatientInfo, value: string) => {
    setPatientInfo(prev => ({
      ...prev,
      [field]: value || ''  // Ensure empty string if value is null/undefined
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Check if at least one symptom is selected
    const hasSymptoms = Object.values(symptoms).some(val => val === true);
    if (!hasSymptoms) {
      setError('Please select at least one symptom before predicting.');
      return;
    }
    
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/predict';
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symptoms,
          patient_info: patientInfo,
        }),
      });

      if (!response.ok) {
        let errorText = 'Unknown error';
        try {
          errorText = await response.text();
        } catch {
          errorText = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(`Prediction failed: ${errorText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred. Please check if the backend server is running.';
      setError(errorMessage);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSymptoms({
      coryza: false,
      wheezing: false,
      sore_throat: false,
      nasal_congestion: false,
      fever: false,
      cough: false,
      chills: false,
      difficulty_breathing: false,
      coughing_up_sputum: false,
      shortness_of_breath: false,
      chest_tightness: false,
      vomiting: false,
    });
    setPatientInfo({
      name: '',
      age: '',
      education: '',
      religion: '',
    });
    setResult(null);
    setError(null);
  };

  const symptomLabels: { [key in keyof Symptoms]: string } = {
    coryza: 'Coryza (Runny Nose)',
    wheezing: 'Wheezing',
    sore_throat: 'Sore Throat',
    nasal_congestion: 'Nasal Congestion',
    fever: 'Fever',
    cough: 'Cough',
    chills: 'Chills',
    difficulty_breathing: 'Difficulty Breathing',
    coughing_up_sputum: 'Coughing Up Sputum',
    shortness_of_breath: 'Shortness of Breath',
    chest_tightness: 'Chest Tightness',
    vomiting: 'Vomiting',
  };

  return (
    <div style={styles.container}>
      <div style={styles.content}>
        <h1 style={styles.title}>ARI Prediction System</h1>
        <p style={styles.subtitle}>Acute Respiratory Infection Risk Assessment</p>

        <form onSubmit={handleSubmit} style={styles.form}>
          {/* Patient Information Section */}
          <div style={styles.section}>
            <h2 style={styles.sectionTitle}>Patient Information</h2>
            <div style={styles.inputGrid}>
              <div style={styles.inputGroup}>
                <label style={styles.label}>Name</label>
                <input
                  type="text"
                  value={patientInfo.name}
                  onChange={(e) => handlePatientInfoChange('name', e.target.value)}
                  style={styles.input}
                  placeholder="Enter patient name"
                />
              </div>
              <div style={styles.inputGroup}>
                <label style={styles.label}>Age</label>
                <input
                  type="number"
                  value={patientInfo.age || ''}
                  onChange={(e) => handlePatientInfoChange('age', e.target.value)}
                  style={styles.input}
                  placeholder="Enter age"
                  min="0"
                />
              </div>
              <div style={styles.inputGroup}>
                <label style={styles.label}>Education</label>
                <input
                  type="text"
                  value={patientInfo.education}
                  onChange={(e) => handlePatientInfoChange('education', e.target.value)}
                  style={styles.input}
                  placeholder="Enter education level"
                />
              </div>
              <div style={styles.inputGroup}>
                <label style={styles.label}>Religion</label>
                <input
                  type="text"
                  value={patientInfo.religion}
                  onChange={(e) => handlePatientInfoChange('religion', e.target.value)}
                  style={styles.input}
                  placeholder="Enter religion"
                />
              </div>
            </div>
          </div>

          {/* Symptoms Section */}
          <div style={styles.section}>
            <h2 style={styles.sectionTitle}>Symptoms Checklist</h2>
            <p style={styles.sectionDescription}>Check all symptoms that apply:</p>
            <div style={styles.checkboxGrid}>
              {Object.keys(symptoms).map((symptom) => (
                <label key={symptom} style={styles.checkboxLabel}>
                  <input
                    type="checkbox"
                    checked={symptoms[symptom as keyof Symptoms]}
                    onChange={() => handleSymptomChange(symptom as keyof Symptoms)}
                    style={styles.checkbox}
                  />
                  <span>{symptomLabels[symptom as keyof Symptoms]}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Action Buttons */}
          <div style={styles.buttonGroup}>
            <button
              type="submit"
              disabled={loading}
              style={loading ? { ...styles.submitButton, ...styles.buttonDisabled } : styles.submitButton}
            >
              {loading ? 'Predicting...' : 'Predict ARI'}
            </button>
            <button
              type="button"
              onClick={handleClear}
              style={styles.clearButton}
            >
              Clear All
            </button>
          </div>
        </form>

        {/* Results Section */}
        {error && (
          <div style={styles.errorBox}>
            <p style={styles.errorText}>Error: {error}</p>
          </div>
        )}

        {result && (
          <div style={styles.resultBox}>
            <h2 style={styles.resultTitle}>Prediction Result</h2>
            <div style={styles.resultContent}>
              <div style={styles.resultItem}>
                <span style={styles.resultLabel}>Prediction:</span>
                <span style={{
                  ...styles.resultValue,
                  color: result.prediction === 'ARI' ? '#dc2626' : '#16a34a'
                }}>
                  {result.prediction === 'ARI' ? 'Positive (ARI Detected)' : 'Negative (No ARI)'}
                </span>
              </div>
              <div style={styles.resultItem}>
                <span style={styles.resultLabel}>Probability:</span>
                <span style={styles.resultValue}>
                  {(result.probability * 100).toFixed(2)}%
                </span>
              </div>
              <div style={styles.resultItem}>
                <span style={styles.resultLabel}>Confidence:</span>
                <span style={styles.resultValue}>{result.confidence}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    minHeight: '100vh',
    backgroundColor: '#f3f4f6',
    padding: '20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  },
  content: {
    maxWidth: '900px',
    margin: '0 auto',
    backgroundColor: 'white',
    borderRadius: '12px',
    padding: '40px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  },
  title: {
    fontSize: '32px',
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: '8px',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: '16px',
    color: '#6b7280',
    textAlign: 'center',
    marginBottom: '32px',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '32px',
  },
  section: {
    border: '1px solid #e5e7eb',
    borderRadius: '8px',
    padding: '24px',
    backgroundColor: '#f9fafb',
  },
  sectionTitle: {
    fontSize: '20px',
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: '8px',
  },
  sectionDescription: {
    fontSize: '14px',
    color: '#6b7280',
    marginBottom: '16px',
  },
  inputGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '16px',
  },
  inputGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  label: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#374151',
  },
  input: {
    padding: '10px 12px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
    outline: 'none',
    transition: 'border-color 0.2s',
  },
  checkboxGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '12px',
  },
  checkboxLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    cursor: 'pointer',
    padding: '8px',
    borderRadius: '6px',
    transition: 'background-color 0.2s',
  },
  checkbox: {
    width: '18px',
    height: '18px',
    cursor: 'pointer',
  },
  buttonGroup: {
    display: 'flex',
    gap: '12px',
    justifyContent: 'center',
  },
  submitButton: {
    padding: '12px 32px',
    backgroundColor: '#3b82f6',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    fontSize: '16px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  buttonDisabled: {
    backgroundColor: '#9ca3af',
    cursor: 'not-allowed',
  },
  clearButton: {
    padding: '12px 32px',
    backgroundColor: '#6b7280',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    fontSize: '16px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  errorBox: {
    marginTop: '24px',
    padding: '16px',
    backgroundColor: '#fef2f2',
    border: '1px solid #fecaca',
    borderRadius: '8px',
  },
  errorText: {
    color: '#dc2626',
    fontSize: '14px',
  },
  resultBox: {
    marginTop: '24px',
    padding: '24px',
    backgroundColor: '#f0f9ff',
    border: '2px solid #3b82f6',
    borderRadius: '8px',
  },
  resultTitle: {
    fontSize: '20px',
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: '16px',
  },
  resultContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  resultItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '8px 0',
    borderBottom: '1px solid #e5e7eb',
  },
  resultLabel: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#6b7280',
  },
  resultValue: {
    fontSize: '16px',
    fontWeight: '600',
    color: '#1f2937',
  },
};

