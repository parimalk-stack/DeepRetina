Dropzone.autoDiscover = false;

// Global variables
let currentBatchResults = [];

// Get CSRF token from meta tag
const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content;

// Initialize Dropzone instances
const singleDropzone = new Dropzone("#singleDropzone", {
    url: "/upload",
    maxFiles: 1,
    acceptedFiles: "image/*",
    autoProcessQueue: false,
    addRemoveLinks: true,
    dictDefaultMessage: "Drop an OCT image here or click to upload",
    dictRemoveFile: "Remove",
    dictCancelUpload: "Cancel",
    maxFilesize: 16, // 16MB
    timeout: 300000, // 5 minutes
    headers: {
        'X-CSRF-TOKEN': csrfToken
    }
});

const compareDropzone = new Dropzone("#compareDropzone", {
    url: "/compare",
    maxFiles: 1,
    acceptedFiles: "image/*",
    autoProcessQueue: false,
    addRemoveLinks: true,
    dictDefaultMessage: "Drop an OCT image here or click to upload",
    dictRemoveFile: "Remove",
    dictCancelUpload: "Cancel",
    maxFilesize: 16,
    timeout: 300000,
    headers: {
        'X-CSRF-TOKEN': csrfToken
    }
});

const qualityDropzone = new Dropzone("#qualityDropzone", {
    url: "/quality_assessment",
    maxFiles: 1,
    acceptedFiles: "image/*",
    autoProcessQueue: false,
    addRemoveLinks: true,
    dictDefaultMessage: "Drop an OCT image here or click to upload",
    dictRemoveFile: "Remove",
    dictCancelUpload: "Cancel",
    maxFilesize: 16,
    timeout: 300000,
    headers: {
        'X-CSRF-TOKEN': csrfToken
    }
});

const batchDropzone = new Dropzone("#batchDropzone", {
    url: "/batch_analyze",
    paramName: "files",
    acceptedFiles: "image/*",
    autoProcessQueue: false,
    addRemoveLinks: true,
    dictDefaultMessage: "Drop OCT images here or click to upload",
    dictRemoveFile: "Remove",
    dictCancelUpload: "Cancel",
    maxFilesize: 16,
    timeout: 300000,
    headers: {
        'X-CSRF-TOKEN': csrfToken
    }
});

// Event Listeners
document.getElementById("analyzeBtn").addEventListener("click", function() {
    if (singleDropzone.getQueuedFiles().length === 0) {
        alert("Please upload an image first.");
        return;
    }
    document.getElementById("loadingOverlay").style.display = "flex";
    singleDropzone.processQueue();
});

document.getElementById("compareBtn").addEventListener("click", function() {
    if (compareDropzone.getQueuedFiles().length === 0) {
        alert("Please upload an image first.");
        return;
    }
    
    // Get selected models
    const modelCheckboxes = document.querySelectorAll('input[name="models"]:checked');
    const selectedModels = Array.from(modelCheckboxes).map(cb => cb.value);
    
    if (selectedModels.length === 0) {
        alert("Please select at least one model to compare.");
        return;
    }
    
    console.log("Selected models:", selectedModels); // Debug log
    
    document.getElementById("loadingOverlay").style.display = "flex";
    
    // Create FormData and send via fetch instead of dropzone
    const files = compareDropzone.getQueuedFiles();
    const formData = new FormData();
    
    // Add the file
    formData.append('file', files[0]);
    
    // Add selected models
    selectedModels.forEach(model => {
        formData.append('models', model);
    });
    
    // Add CSRF token
    formData.append('csrf_token', csrfToken);
    
    // Send request via fetch
    fetch('/compare', {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRF-TOKEN': csrfToken
        }
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("loadingOverlay").style.display = "none";
        if (data.error) {
            alert("Error: " + data.error);
            return;
        }
        displayComparisonResults(data);
    })
    .catch(error => {
        document.getElementById("loadingOverlay").style.display = "none";
        console.error('Error during comparison:', error);
        alert('An error occurred during model comparison.');
    });
});

document.getElementById("assessQualityBtn").addEventListener("click", function() {
    if (qualityDropzone.getQueuedFiles().length === 0) {
        alert("Please upload an image first.");
        return;
    }
    document.getElementById("loadingOverlay").style.display = "flex";
    qualityDropzone.processQueue();
});

document.getElementById("batchAnalyzeBtn").addEventListener("click", function() {
    const files = batchDropzone.getQueuedFiles();
    if (files.length === 0) {
        alert("Please upload images first.");
        return;
    }

    document.getElementById("loadingOverlay").style.display = "flex";

    const formData = new FormData();
    files.forEach(function(file) {
        formData.append('files[]', file);
    });

    // Get selected model
    const modelSelect = document.getElementById('batchModelSelect');
    if (modelSelect) { // Check if modelSelect exists for disease classification tab
        const modelName = modelSelect.value;
        formData.append('model', modelName);
    }
    
    // Get analysis type
    const analysisTypeSelect = document.getElementById('batchAnalysisType');
    const analysisType = analysisTypeSelect.value;
    formData.append('analysis_type', analysisType);

    // Add CSRF token
    formData.append('csrf_token', csrfToken);

    // Determine the correct URL based on analysis type
    const uploadUrl = analysisType === 'quality_assessment' ? '/batch_quality_assessment' : '/batch_analyze';

    fetch(uploadUrl, {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRF-TOKEN': csrfToken // Also include in headers for good measure
        }
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("loadingOverlay").style.display = "none";
        if (data.error) {
            alert("Error: " + data.error);
            // Optionally remove files from Dropzone on error
            // batchDropzone.removeAllFiles();
        } else {
            // Handle successful batch analysis response
            if (analysisType === 'quality_assessment') {
                 displayBatchQualityResults(data); // Assuming you have a function for quality batch results
            } else {
                 displayBatchResults(data);
            }
           
            // Clear Dropzone queue after successful upload
            batchDropzone.removeAllFiles();
        }
    })
    .catch(error => {
        document.getElementById("loadingOverlay").style.display = "none";
        console.error('Error during batch analysis:', error);
        alert('An error occurred during batch analysis.');
        // Optionally remove files from Dropzone on error
        // batchDropzone.removeAllFiles();
    });
});

// Analysis type change handler
document.getElementById("batchAnalysisType").addEventListener("change", function(e) {
    const modelSelection = document.getElementById("batchModelSelection");
    if (e.target.value === "quality_assessment") {
        modelSelection.style.display = "none";
    } else {
        modelSelection.style.display = "block";
    }
});

// Compare models form submission - this now just triggers the button click
document.getElementById("compareModelsForm").addEventListener("submit", function(e) {
    e.preventDefault();
    
    // Trigger the compare button click which has the main logic
    document.getElementById("compareBtn").click();
});

// Handle single analysis response
singleDropzone.on("success", function(file, response) {
    // Hide loading overlay
    document.getElementById("loadingOverlay").style.display = "none";
    
    if (response.error) {
        alert("Error: " + response.error);
        return;
    }
    
    displaySingleResult(response);
});

// Handle comparison response - now handled by fetch in the button click
// This is kept for compatibility but may not be used
compareDropzone.on("success", function(file, response) {
    console.log("Dropzone success handler (may not be used):", response);
});

// Handle quality assessment response
qualityDropzone.on("success", function(file, response) {
    // Hide loading overlay
    document.getElementById("loadingOverlay").style.display = "none";
    
    if (response.error) {
        alert("Error: " + response.error);
        return;
    }
    
    displayQualityResults(response);
});

// Handle batch analysis response
batchDropzone.on("success", function(file, response) {
    // Hide loading overlay
    document.getElementById("loadingOverlay").style.display = "none";
    
    if (response.error) {
        alert("Error: " + response.error);
        return;
    }
    
    if (response.results) {
        currentBatchResults = response.results;
        displayBatchResults(response);
    } else {
        displayBatchQualityResults(response);
    }
});

// Error handlers
[singleDropzone, compareDropzone, qualityDropzone, batchDropzone].forEach(dropzone => {
    dropzone.on("error", function(file, errorMessage) {
        document.getElementById("loadingOverlay").style.display = "none";
        alert("Error: " + errorMessage);
    });
});

// Display functions
function displaySingleResult(response) {
    document.getElementById("singleResultContainer").style.display = "block";
    
    const timestamp = new Date().getTime();
    document.getElementById("originalImage").src = `/static/uploads/${response.filename}?${timestamp}`;
    document.getElementById("resultImage").src = `/static/${response.result.visualization}?${timestamp}`;
    
    document.getElementById("predictionResult").textContent = response.result.predicted_class;
    document.getElementById("confidenceValue").textContent = (response.result.confidence * 100).toFixed(2);
    document.getElementById("processingTime").textContent = (response.result.processing_time + response.result.inference_time).toFixed(3);
    
    const probs = response.result.all_probabilities;
    
    document.getElementById("cnvProb").textContent = (probs.CNV * 100).toFixed(2) + "%";
    document.getElementById("cnvBar").style.width = (probs.CNV * 100) + "%";
    
    document.getElementById("dmeProb").textContent = (probs.DME * 100).toFixed(2) + "%";
    document.getElementById("dmeBar").style.width = (probs.DME * 100) + "%";
    
    document.getElementById("drusenProb").textContent = (probs.DRUSEN * 100).toFixed(2) + "%";
    document.getElementById("drusenBar").style.width = (probs.DRUSEN * 100) + "%";
    
    document.getElementById("normalProb").textContent = (probs.NORMAL * 100).toFixed(2) + "%";
    document.getElementById("normalBar").style.width = (probs.NORMAL * 100) + "%";
}

function displayComparisonResults(response) {
    document.getElementById("compareResultContainer").style.display = "block";
    
    const timestamp = new Date().getTime();
    document.getElementById("compareOriginalImage").src = `/static/uploads/${response.filename}?${timestamp}`;
    
    const modelResults = document.getElementById("modelResults");
    modelResults.innerHTML = "";
    
    console.log("Displaying comparison results for models:", Object.keys(response.results));
    
    // Model display name mapping
    const modelDisplayNames = {
        'cnn_model': 'Custom CNN',
        'vit_model': 'Vision Transformer',
        'swin_model': 'Swin Transformer'
    };
    
    Object.keys(response.results).forEach(model => {
        const result = response.results[model];
        const displayName = modelDisplayNames[model] || model;
        
        // Determine card width based on number of models
        const numModels = Object.keys(response.results).length;
        const cardClass = numModels === 1 ? 'col-md-12' : numModels === 2 ? 'col-md-6' : 'col-md-4';
        
        const cardDiv = document.createElement('div');
        cardDiv.className = `${cardClass} mb-4`;
        cardDiv.innerHTML = `
            <div class="card h-100">
                <div class="card-header bg-primary text-white">
                    <h5 class="mb-0">${displayName}</h5>
                    <small>${model}</small>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <h6>Visualization</h6>
                            <img src="/static/${result.visualization}?${timestamp}" class="img-fluid mb-3" alt="${model} result">
                        </div>
                        <div class="col-md-6">
                            <h6>Prediction Results</h6>
                            <div class="alert alert-info">
                                <strong>Predicted:</strong> ${result.predicted_class}<br>
                                <strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%<br>
                                <strong>Processing Time:</strong> ${(result.processing_time + result.inference_time).toFixed(3)}s
                            </div>
                            
                            <h6>Class Probabilities</h6>
                            <div class="mb-2">
                                <div class="d-flex justify-content-between">
                                    <span>CNV</span>
                                    <span><strong>${(result.all_probabilities.CNV * 100).toFixed(2)}%</strong></span>
                                </div>
                                <div class="progress mb-1">
                                    <div class="progress-bar bg-danger" role="progressbar" style="width: ${result.all_probabilities.CNV * 100}%"></div>
                                </div>
                            </div>
                            <div class="mb-2">
                                <div class="d-flex justify-content-between">
                                    <span>DME</span>
                                    <span><strong>${(result.all_probabilities.DME * 100).toFixed(2)}%</strong></span>
                                </div>
                                <div class="progress mb-1">
                                    <div class="progress-bar bg-warning" role="progressbar" style="width: ${result.all_probabilities.DME * 100}%"></div>
                                </div>
                            </div>
                            <div class="mb-2">
                                <div class="d-flex justify-content-between">
                                    <span>DRUSEN</span>
                                    <span><strong>${(result.all_probabilities.DRUSEN * 100).toFixed(2)}%</strong></span>
                                </div>
                                <div class="progress mb-1">
                                    <div class="progress-bar bg-info" role="progressbar" style="width: ${result.all_probabilities.DRUSEN * 100}%"></div>
                                </div>
                            </div>
                            <div class="mb-2">
                                <div class="d-flex justify-content-between">
                                    <span>NORMAL</span>
                                    <span><strong>${(result.all_probabilities.NORMAL * 100).toFixed(2)}%</strong></span>
                                </div>
                                <div class="progress mb-1">
                                    <div class="progress-bar bg-success" role="progressbar" style="width: ${result.all_probabilities.NORMAL * 100}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        modelResults.appendChild(cardDiv);
    });
    
    console.log("Finished displaying results for", Object.keys(response.results).length, "models");
}

function displayQualityResults(response) {
    document.getElementById("qualityResultContainer").style.display = "block";
    
    const timestamp = new Date().getTime();
    document.getElementById("qualityOriginalImage").src = `/static/uploads/${response.filename}?${timestamp}`;
    document.getElementById("qualityVisualization").src = `/static/${response.visualization}?${timestamp}`;
    
    document.getElementById("qualityScore").textContent = response.quality_score.toFixed(2);
    document.getElementById("qualitySNR").textContent = response.snr.toFixed(2);
    document.getElementById("qualityMotion").textContent = response.motion_artifacts.toFixed(2);
    document.getElementById("qualityContrast").textContent = response.contrast.toFixed(2);
    document.getElementById("qualityBlur").textContent = response.blur_metric.toFixed(2);
    
    const statusBadge = document.getElementById("qualityStatusBadge");
    statusBadge.textContent = response.status;
    statusBadge.className = "badge " + (response.status === "Good" ? "bg-success" : "bg-warning");
    
    const recommendationsList = document.getElementById("qualityRecommendations");
    recommendationsList.innerHTML = "";
    response.recommendations.forEach(rec => {
        const li = document.createElement("li");
        li.textContent = rec;
        recommendationsList.appendChild(li);
    });
}

function displayBatchResults(response) {
    document.getElementById("batchResultContainer").style.display = "block";
    
    // Update summary metrics
    document.getElementById("batchTotalFiles").textContent = response.total_processed;
    document.getElementById("batchProcessedFiles").textContent = response.total_processed;
    document.getElementById("batchAvgQuality").textContent = response.metrics.average_confidence.toFixed(2);
    document.getElementById("batchProcessingTime").textContent = response.metrics.total_processing_time.toFixed(2) + "s";
    
    // Display visualizations
    const vizContainer = document.getElementById("batchVisualizations");
    vizContainer.innerHTML = "";
    
    if (response.visualizations) {
        Object.entries(response.visualizations).forEach(([name, path]) => {
            const colDiv = document.createElement('div');
            colDiv.className = 'col-md-6 mb-4';
            colDiv.innerHTML = `
                <div class="card">
                    <div class="card-header">${name}</div>
                    <div class="card-body">
                        <img src="/static/${path}" class="img-fluid" alt="${name}">
                    </div>
                </div>
            `;
            vizContainer.appendChild(colDiv);
        });
    }
    
    // Display detailed results
    const resultsContainer = document.getElementById("batchDetailedResults");
    resultsContainer.innerHTML = "";
    
    response.results.forEach(result => {
        const resultDiv = document.createElement('div');
        resultDiv.className = `batch-result-item ${result.correct === true ? 'correct' : result.correct === false ? 'incorrect' : 'unknown'}`;
        
        resultDiv.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <strong>File:</strong> ${result.filename}
                </div>
                <div class="col-md-3">
                    <strong>Prediction:</strong> ${result.predicted_class}
                </div>
                <div class="col-md-2">
                    <strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%
                </div>
                <div class="col-md-2">
                    <strong>Time:</strong> ${(result.processing_time + result.inference_time).toFixed(3)}s
                </div>
                <div class="col-md-2">
                    <strong>Status:</strong> 
                    ${result.correct === true ? 
                        '<span class="badge bg-success">Correct</span>' : 
                        result.correct === false ? 
                            '<span class="badge bg-danger">Incorrect</span>' : 
                            '<span class="badge bg-secondary">Unknown</span>'}
                </div>
            </div>
        `;
        
        resultsContainer.appendChild(resultDiv);
    });
    
    // Enable export buttons
    document.getElementById("exportBatchCSV").disabled = false;
    document.getElementById("exportBatchReport").disabled = false;
}

function displayBatchQualityResults(response) {
    document.getElementById("batchResultContainer").style.display = "block";
    
    // Update summary metrics
    document.getElementById("batchTotalFiles").textContent = response.batch_statistics.total_files;
    document.getElementById("batchProcessedFiles").textContent = response.total_processed;
    document.getElementById("batchAvgQuality").textContent = response.batch_statistics.average_quality_score.toFixed(2);
    document.getElementById("batchProcessingTime").textContent = "N/A";
    
    // Display detailed results
    const resultsContainer = document.getElementById("batchDetailedResults");
    resultsContainer.innerHTML = "";
    
    response.results.forEach(result => {
        const resultDiv = document.createElement('div');
        resultDiv.className = `batch-result-item ${result.processable ? 'correct' : 'incorrect'}`;
        
        resultDiv.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <strong>File:</strong> ${result.original_filename}
                </div>
                <div class="col-md-3">
                    <strong>Quality Score:</strong> ${result.overall_quality_score.toFixed(2)}
                </div>
                <div class="col-md-3">
                    <strong>SNR:</strong> ${result.snr_db.toFixed(2)} dB
                </div>
                <div class="col-md-3">
                    <strong>Status:</strong> 
                    ${result.processable ? 
                        '<span class="badge bg-success">Good</span>' : 
                        '<span class="badge bg-danger">Poor</span>'}
                </div>
            </div>
        `;
        
        resultsContainer.appendChild(resultDiv);
    });
    
    // Enable export buttons
    document.getElementById("exportBatchCSV").disabled = false;
    document.getElementById("exportBatchReport").disabled = false;
}

// Export functions
document.getElementById("exportBatchCSV").addEventListener("click", function() {
    exportBatchResults();
});

document.getElementById("exportBatchReport").addEventListener("click", function() {
    exportBatchReport();
});

function exportBatchResults() {
    if (currentBatchResults.length === 0) {
        alert('No batch results to export.');
        return;
    }
    
    fetch('/export_results', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-TOKEN': csrfToken
        },
        body: JSON.stringify({
            type: 'batch',
            data: currentBatchResults
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        downloadFile(data.csv_data, data.filename, 'text/csv');
    })
    .catch(error => {
        console.error('Error exporting batch results:', error);
        alert('Error exporting batch results.');
    });
}

function exportBatchReport() {
    if (currentBatchResults.length === 0) {
        alert('No batch results to export.');
        return;
    }
    
    fetch('/export_report', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-TOKEN': csrfToken
        },
        body: JSON.stringify({
            data: currentBatchResults
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        downloadFile(data.html_data, data.filename, 'text/html');
    })
    .catch(error => {
        console.error('Error exporting report:', error);
        alert('Error exporting report.');
    });
}

function downloadFile(content, filename, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

// Reset buttons
document.getElementById("newImageButton").addEventListener("click", function() {
    document.getElementById("singleResultContainer").style.display = "none";
    singleDropzone.removeAllFiles();
    document.getElementById("analyzeBtn").disabled = true;
});

document.getElementById("newCompareButton").addEventListener("click", function() {
    document.getElementById("compareResultContainer").style.display = "none";
    compareDropzone.removeAllFiles();
    document.getElementById("compareBtn").disabled = true;
});

document.getElementById("newQualityButton").addEventListener("click", function() {
    document.getElementById("qualityResultContainer").style.display = "none";
    qualityDropzone.removeAllFiles();
    document.getElementById("assessQualityBtn").disabled = true;
});

document.getElementById("newBatchButton").addEventListener("click", function() {
    document.getElementById("batchResultContainer").style.display = "none";
    batchDropzone.removeAllFiles();
    document.getElementById("batchAnalyzeBtn").disabled = true;
    currentBatchResults = [];
});

// Model info display
document.getElementById("modelSelect").addEventListener("change", function() {
    const selectedOption = this.options[this.selectedIndex];
    const architecture = selectedOption.dataset.architecture;
    const parameters = selectedOption.dataset.parameters;
    document.getElementById("modelInfo").textContent = `Architecture: ${architecture}, Parameters: ${parameters}`;
});

// Initialize on page load
window.addEventListener("DOMContentLoaded", function() {
    const modelSelect = document.getElementById("modelSelect");
    if (modelSelect.options.length > 0) {
        modelSelect.dispatchEvent(new Event('change'));
    }
});

// Single Dropzone Listeners
singleDropzone.on("addedfile", function() {
    document.getElementById("analyzeBtn").disabled = false;
});

singleDropzone.on("removedfile", function() {
    if (singleDropzone.getQueuedFiles().length === 0) {
        document.getElementById("analyzeBtn").disabled = true;
    }
});

// Compare Dropzone Listeners
compareDropzone.on("addedfile", function() {
    document.getElementById("compareBtn").disabled = false;
});

compareDropzone.on("removedfile", function() {
    if (compareDropzone.getQueuedFiles().length === 0) {
        document.getElementById("compareBtn").disabled = true;
    }
});

// Quality Dropzone Listeners
qualityDropzone.on("addedfile", function() {
    document.getElementById("assessQualityBtn").disabled = false;
});

qualityDropzone.on("removedfile", function() {
    if (qualityDropzone.getQueuedFiles().length === 0) {
        document.getElementById("assessQualityBtn").disabled = true;
    }
});

// Batch Dropzone Listeners
batchDropzone.on("addedfile", function() {
    document.getElementById("batchAnalyzeBtn").disabled = false;
});

batchDropzone.on("removedfile", function() {
    if (batchDropzone.getQueuedFiles().length === 0) {
        document.getElementById("batchAnalyzeBtn").disabled = true;
    }
}); 