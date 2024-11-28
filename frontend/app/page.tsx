"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { X, ArrowLeft, Upload, ChevronRight } from "lucide-react";
import Script from 'next/script';

const TextEmbeddingVisualizer = () => {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [selectedColumn, setSelectedColumn] = useState('');
  const [plotData, setPlotData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [plotlyReady, setPlotlyReady] = useState(false);
  const [selectedText, setSelectedText] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [stage, setStage] = useState('upload'); // 'upload', 'select', 'visualize'
  const [isTransitioning, setIsTransitioning] = useState(false);

  useEffect(() => {
    if (plotlyReady && plotData) {
      renderPlot();
    }
  }, [plotData, plotlyReady]);

  const renderPlot = () => {
    if (!plotData || !window.Plotly) return;

    const data = [];
    
    // Add density heatmaps first (background layer)
    // Add density heatmaps with reduced visibility
    Object.entries(plotData.topic_centers).forEach(([topicId, center]) => {
      data.push({
          type: 'contour',
          x: center.density_map.x,
          y: center.density_map.y,
          z: center.density_map.z,
          contours: {
              coloring: 'fill',
              showlines: false,
              // Reduce number of contour levels for less intensive coloring
              size: center.density_map.max_density / 10,  // Fewer contour levels
              start: center.density_map.max_density * 0.2,  // Start at higher threshold
              end: center.density_map.max_density
          },
          colorscale: [
              [0, 'rgba(255,255,255,0)'],
              // Adjust the second number (0.05) to control transparency
              // Lower = more transparent. Range: 0.01 to 0.15
              [1, center.color.replace('rgb', 'rgba').replace(')', ',0.05)')],
          ],
          showscale: false,
          name: `Topic ${topicId}: ${center.keywords}`,
          hoverongaps: false,
          showlegend: false,
          // Reduce overall opacity of the contour layer
          opacity: 0.3  // Adjust this value (0.1 to 0.5) to control overall visibility
      });
    });

    // Add scatter points (middle layer)
    Object.entries(plotData.topic_centers).forEach(([topicId, center]) => {
        const topicPoints = plotData.topics.map((t, i) => t === parseInt(topicId));
        data.push({
            type: 'scatter',
            x: plotData.x.filter((_, i) => topicPoints[i]),
            y: plotData.y.filter((_, i) => topicPoints[i]),
            text: plotData.texts.filter((_, i) => topicPoints[i]),
            mode: 'markers',
            name: `Topic ${topicId}: ${center.keywords}`,  // Updated name format
            marker: {
                size: 8,
                color: center.color,
                opacity: 0.8,
                line: {
                    color: 'white',
                    width: 1
                }
            },
            hovertemplate: 
                "<b>Topic:</b> %{data.name}<br>" +
                "<b>Text:</b> %{text}<br>" +
                "<extra></extra>"
        });
    });

    // Add enhanced topic labels with numbers (top layer)
    Object.entries(plotData.topic_centers).forEach(([topicId, center]) => {
        const labelText = `Topic ${topicId}\n${center.keywords}`;  // Two-line format
        
        data.push({
            type: 'scatter',
            x: [center.x],
            y: [center.y],
            mode: 'text',
            text: [labelText],
            textfont: {
                size: 14,
                color: 'rgba(0,0,0,0.85)',
                weight: 600,
                family: 'Inter, sans-serif'
            },
            textposition: 'middle center',
            showlegend: false,
            hoverinfo: 'none'
        });
    });

    const layout = {
        title: 'Topic Clusters Visualization',
        plot_bgcolor: '#fafafa',
        paper_bgcolor: 'white',
        autosize: true,
        height: 800,
        hovermode: 'closest',
        margin: { l: 40, r: 40, t: 60, b: 40 },
        legend: {
            yanchor: "top",
            y: 0.99,
            xanchor: "left",
            x: 0.01,
            bgcolor: 'rgba(255, 255, 255, 0.95)',
            bordercolor: 'rgba(0,0,0,0.1)',
            borderwidth: 1
        },
        xaxis: {
            title: "Dimension 1",
            gridcolor: '#eee',
            showgrid: true
        },
        yaxis: {
            title: "Dimension 2",
            gridcolor: '#eee',
            showgrid: true
        },
        // Enhanced annotations with topic numbers
        annotations: Object.entries(plotData.topic_centers).map(([topicId, center]) => ({
            x: center.x,
            y: center.y,
            text: `Topic ${topicId}:\n${center.keywords}`,  // Two-line format
            showarrow: false,
            font: {
                family: 'Inter, sans-serif',
                size: 14,
                color: 'rgba(0,0,0,0.85)',
                weight: 600
            },
            bgcolor: 'rgba(255,255,255,0.9)',
            bordercolor: 'rgba(0,0,0,0.1)',
            borderwidth: 0.1,
            borderpad: 4,
            opacity: 1,
            align: 'center'
        }))
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToAdd: ['resetScale2d'],
        displaylogo: false
    };

    const plot = document.getElementById('plotly-container');
    window.Plotly.newPlot(plot, data, layout, config);

    // Update click handling to include topic number
    plot.on('plotly_click', (data) => {
        const point = data.points[0];
        const text = point.text;
        const topicName = point.data.name;  // Now includes topic number
        const index = plotData.texts.indexOf(text);
        
        setSelectedText({
            text: text,
            topic: topicName,  // Will now show "Topic X: keywords"
            index: index + 1
        });
    });

    // Responsive handling
    const handleResize = () => {
        window.Plotly.Plots.resize(plot);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  };

  const handleFileUpload = async (event) => {
    try {
      setError('');
      const file = event.target.files[0];
      setFile(file);
      
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target.result;
        const headers = text.split('\n')[0].split(',').map(h => h.trim());
        setColumns(headers);
        setStage('select');
      };
      reader.readAsText(file);
    } catch (err) {
      setError('Error reading file. Please try again.');
      console.error('Error:', err);
    }
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError('');
    setSelectedText(null);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('text_column', selectedColumn);

    try {
      const response = await fetch('http://127.0.0.1:5000/process', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to process data');
      }
      
      const result = await response.json();
      
      if (!result.plot_data) {
        throw new Error('No visualization data received');
      }

      setPlotData(result.plot_data);
      setStage('visualize');
    } catch (err) {
      setError(err.message || 'Error processing data. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleStartOver = () => {
    setIsTransitioning(true);
    setTimeout(() => {
      setFile(null);
      setColumns([]);
      setSelectedColumn('');
      setPlotData(null);
      setSelectedText(null);
      setStage('upload');
      setIsTransitioning(false);
    }, 300);
  };

  return (
    <>
      <Script 
        src="https://cdn.plot.ly/plotly-2.27.0.min.js"
        onLoad={() => setPlotlyReady(true)}
      />
      
      <div className="min-h-screen bg-gray-50 p-4">
        {/* Main Container with stage-based transitions */}
        <div className={`
          transition-all duration-500 ease-in-out
          ${stage === 'upload' ? 'max-w-md' : stage === 'select' ? 'max-w-xl' : 'max-w-[90vw]'}
          mx-auto
        `}>
          {/* Upload and Select Stages */}
          {(stage === 'upload' || stage === 'select') && (
            <Card className={`
              transition-all duration-500 ease-in-out transform
              ${isTransitioning ? 'scale-95 opacity-0' : 'scale-100 opacity-100'}
            `}>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  {stage === 'select' && (
                    <Button variant="ghost" size="sm" onClick={() => setStage('upload')}>
                      <ArrowLeft className="h-4 w-4" />
                    </Button>
                  )}
                  <span>Text Embedding Visualizer</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {error && (
                    <div className="bg-red-50 border border-red-200 text-red-600 px-4 py-2 rounded-lg">
                      {error}
                    </div>
                  )}
                  
                  {stage === 'upload' && (
                    <div className="space-y-4">
                      <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                        <Upload className="h-8 w-8 mx-auto mb-4 text-gray-400" />
                        <label className="block text-sm font-medium mb-2 cursor-pointer">
                          <span className="text-blue-500 hover:text-blue-600">Upload CSV File</span>
                          <input
                            type="file"
                            accept=".csv"
                            onChange={handleFileUpload}
                            className="hidden"
                          />
                        </label>
                      </div>
                    </div>
                  )}

                  {stage === 'select' && (
                    <div className="space-y-4">
                      <div className="text-sm text-gray-500">
                        Selected file: {file?.name}
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-2">
                          Select Text Column
                        </label>
                        <select
                          value={selectedColumn}
                          onChange={(e) => setSelectedColumn(e.target.value)}
                          className="block w-full p-2 border rounded-lg"
                        >
                          <option value="">Select a column</option>
                          {columns.map((col) => (
                            <option key={col} value={col}>
                              {col}
                            </option>
                          ))}
                        </select>
                      </div>
                      <Button
                        onClick={handleSubmit}
                        disabled={!selectedColumn || loading}
                        className="w-full"
                      >
                        {loading ? (
                          'Processing...'
                        ) : (
                          <span className="flex items-center justify-center">
                            Generate Visualization
                            <ChevronRight className="ml-2 h-4 w-4" />
                          </span>
                        )}
                      </Button>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Visualization Stage */}
          {stage === 'visualize' && (
            <div className={`
              transition-all duration-500 ease-in-out transform
              ${isTransitioning ? 'scale-95 opacity-0' : 'scale-100 opacity-100'}
            `}>
              <div className="flex justify-between items-center mb-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleStartOver}
                  className="flex items-center"
                >
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Start Over
                </Button>
              </div>
              
              <div className="flex gap-4">
                {/* Visualization Card */}
                <Card className="flex-1">
                  <CardContent className="p-4">
                    <div id="plotly-container" className="w-full h-[600px] bg-white rounded-lg" />
                  </CardContent>
                </Card>

                {/* Text Sidebar */}
                <div className="w-96">
                  <Card className="sticky top-4">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">
                        {selectedText ? `Text #${selectedText.index}` : 'Click a point to view text'}
                      </CardTitle>
                      {selectedText && (
                        <Button 
                          variant="ghost" 
                          size="sm"
                          onClick={() => setSelectedText(null)}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      )}
                    </CardHeader>
                    <CardContent>
                      {selectedText ? (
                        <div className="space-y-4">
                          <div className="text-sm">
                            <span className="font-medium">Topic: </span>
                            <span className="text-gray-600">{selectedText.topic}</span>
                          </div>
                          <div className="text-sm whitespace-pre-wrap">
                            <span className="font-medium">Text: </span>
                            <span className="text-gray-600">{selectedText.text}</span>
                          </div>
                        </div>
                      ) : (
                        <div className="text-sm text-gray-500 mt-2">
                          No text selected. Click on any point in the visualization to view its content.
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default TextEmbeddingVisualizer;