
/* Frontend REST Client for Web-Based ML Model Evaluator */

const API_BASE = "/api";

// US-11: Restore the active dataset ID for this browser session.
let activeDatasetId = sessionStorage.getItem("activeDatasetId");

// Initialize frontend when the page loads.
document.addEventListener("DOMContentLoaded", () => {
    checkBackendHealth();
    setupEventListeners();
    setupLoginForm();
    setupRegistrationForm();
    setupAdminPanel();
    document.getElementById("refresh-history-btn")?.addEventListener("click", loadRunHistory);
    document.getElementById("compare-runs-btn")?.addEventListener("click", compareSelectedRuns);
    restoreSession();
});

// Reusable function for communicating with the backend API.
async function apiRequest(endpoint, options = {}) {
    let response;

    try {
        response = await fetch(`${API_BASE}${endpoint}`, {
            ...options,
            credentials: "same-origin",
            headers: {
                ...options.headers
            }
        });
    } catch {
        throw new Error(
            "Unable to connect to the server. Please check your connection."
        );
    }

    const contentType = response.headers.get("content-type") || "";
    let data = {};

    if (contentType.includes("application/json")) {
        try {
            data = await response.json();
        } catch {
            throw new Error("The backend returned invalid JSON.");
        }
    } else {
        if (!response.ok) {
            throw new Error(
                `The server returned an unexpected response (HTTP ${response.status}).`
            );
        }

        throw new Error("The backend returned a non-JSON response.");
    }

    if (!response.ok) {
        throw new Error(
            data.message ||
            data.error ||
            `API request failed with status ${response.status}`
        );
    }

    return data;
}

// Check whether the backend server is available.
async function checkBackendHealth() {
    const badge = document.getElementById("api-status");
    if (!badge) return;

    try {
        const data = await apiRequest("/health");

        if (data.status === "healthy") {
            badge.className = "badge bg-success me-3";
            badge.innerHTML =
                '<i class="bi bi-circle-fill me-1 small"></i>API Connected';
        } else {
            throw new Error("Backend is not healthy");
        }
    } catch (error) {
        badge.className = "badge bg-warning text-dark me-3";
        badge.innerHTML =
            '<i class="bi bi-exclamation-triangle-fill me-1 small"></i>API Offline';

        console.error("Backend connection failed:", error.message);
    }
}

// Set up frontend event listeners.
function setupEventListeners() {
    // US-03: Frontend-backend communication.
    // US-08 and US-11: Upload or replace a CSV dataset.
    document.getElementById("upload-form")
        ?.addEventListener("submit", uploadDataset);
    document.getElementById("train-form")
        ?.addEventListener("submit", evaluateDataset);
    document.getElementById("split-range")
        ?.addEventListener("input", updateSplitDisplay);
    updateSplitDisplay();


    // US-14: Data Preprocessing form listener.
    document.getElementById("preprocess-form")
        ?.addEventListener("submit", applyPreprocessing);

    // US-15: Validate and track the selected target column.
    const targetSelect = document.getElementById("target-column-select");
    const targetFeedback = document.getElementById("target-column-feedback");
    const trainButton = document.getElementById("train-btn");
    document.getElementById("task-type-select")?.addEventListener("change", () => {
        if (trainButton) trainButton.disabled = !targetSelect?.value;
    });

    if (targetSelect) {
        targetSelect.addEventListener("change", () => {
            const selectedTarget = targetSelect.value;

            const availableColumns = Array.from(targetSelect.options)
                .map((option) => option.value)
                .filter(Boolean);

            const isValidTarget =
                Boolean(activeDatasetId) &&
                availableColumns.includes(selectedTarget);

            targetSelect.setCustomValidity(
                isValidTarget ? "" : "Please select a valid target column."
            );

            if (targetFeedback) {
                targetFeedback.textContent = isValidTarget
                    ? `"${selectedTarget}" selected as the prediction target.`
                    : "Please select a valid target column.";

                targetFeedback.classList.toggle("text-success", isValidTarget);
                targetFeedback.classList.toggle("text-danger", !isValidTarget);
            }

            // Enable the button only when the target is valid.
            if (trainButton) {
                trainButton.disabled = !isValidTarget;
            }
        });
    }

    // US-16: Split ratio range slider listener - DONE (updateSplitDisplay)
    // US-17: Algorithm selection - DONE (loadAlgorithmCatalog)
    // US-18: Model training - DONE (trainModel & backend /train)
    // US-19: Evaluation metrics & comparison - DONE (evaluateDataset)

    const logoutBtn = document.getElementById("logout-btn");

    if (logoutBtn) {
        logoutBtn.addEventListener("click", handleLogout);
    }

    console.log("Frontend architecture initialized.");
}

// US-18: Update training controls based on active dataset state.
function updateTrainingControls(enabled) {
    const targetSelect = document.getElementById("target-column-select");
    const algorithmSelect = document.getElementById("algorithm-select");
    const taskTypeSelect = document.getElementById("task-type-select");
    const splitRange = document.getElementById("split-range");
    const trainBtn = document.getElementById("train-btn");

    if (targetSelect) targetSelect.disabled = !enabled;
    if (algorithmSelect) algorithmSelect.disabled = !enabled;
    if (taskTypeSelect) taskTypeSelect.disabled = !enabled;
    if (splitRange) splitRange.disabled = !enabled;
    if (trainBtn) trainBtn.disabled = !enabled;
}

// US-14: Update preprocessing controls based on active dataset state.
function updatePreprocessingControls(enabled) {
    const missingSelect = document.getElementById("missing-values-select");
    const encodingSelect = document.getElementById("encoding-select");
    const scalingSelect = document.getElementById("scaling-select");
    const preprocessBtn = document.getElementById("preprocess-btn");

    if (missingSelect) missingSelect.disabled = !enabled;
    if (encodingSelect) encodingSelect.disabled = !enabled;
    if (scalingSelect) scalingSelect.disabled = !enabled;
    if (preprocessBtn) preprocessBtn.disabled = !enabled;
}

// US-14: Show user feedback for preprocessing operations.
function showPreprocessFeedback(message, isSuccess = true) {
    const feedback = document.getElementById("preprocess-feedback");
    if (!feedback) return;

    feedback.classList.remove("d-none", "alert-success", "alert-danger", "alert-info");
    feedback.classList.add(isSuccess ? "alert-success" : "alert-danger");
    feedback.textContent = message;
}

// US-14: Apply dataset preprocessing operations.
async function applyPreprocessing(event) {
    event.preventDefault();

    if (!activeDatasetId) {
        showPreprocessFeedback("Upload a dataset before applying preprocessing.", false);
        return;
    }

    const button = document.getElementById("preprocess-btn");
    const missingStrategy = document.getElementById("missing-values-select")?.value || "none";
    const encodingStrategy = document.getElementById("encoding-select")?.value || "none";
    const scalingStrategy = document.getElementById("scaling-select")?.value || "none";
    const targetColumn = document.getElementById("target-column-select")?.value || null;

    if (button) {
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>Applying Preprocessingâ€¦';
    }

    showPreprocessFeedback("Executing preprocessing transformations on datasetâ€¦", true);
    const feedback = document.getElementById("preprocess-feedback");
    if (feedback) {
        feedback.classList.remove("alert-success", "alert-danger");
        feedback.classList.add("alert-info");
    }

    try {
        const payload = {
            missing_values: missingStrategy,
            encoding: encodingStrategy,
            scaling: scalingStrategy,
            target_column: targetColumn,
            save: true
        };

        const result = await apiRequest(`/dataset/${encodeURIComponent(activeDatasetId)}/preprocess`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        updateDatasetDisplay(result);
        showPreprocessFeedback(
            `Preprocessing complete: ${result.row_count} rows and ${result.column_count} columns available.`,
            true
        );

        await loadDatasetPreview(result.dataset_id);
        await loadDatasetStatistics(result.dataset_id);

    } catch (error) {
        showPreprocessFeedback(
            error.message || "Preprocessing configuration could not be applied.",
            false
        );
    } finally {
        if (button) {
            button.disabled = false;
            button.innerHTML = '<i class="bi bi-magic me-1"></i>Apply Preprocessing';
        }
    }
}

// Update the upload button based on active dataset state.
function updateUploadButton() {
    const button = document.getElementById("upload-btn");
    if (!button) return;

    if (activeDatasetId) {
        button.innerHTML =
            '<i class="bi bi-arrow-repeat me-1"></i>Replace Dataset';
    } else {
        button.innerHTML =
            '<i class="bi bi-upload me-1"></i>Upload & Inspect';
    }
}

// Update target-column options using the replacement dataset's columns.
function updateTargetColumnOptions(columns) {
    const select = document.getElementById("target-column-select");
    if (!select) return;

    const targetFeedback = document.getElementById("target-column-feedback");
    const trainButton = document.getElementById("train-btn");

    // Reset the target selection whenever dataset columns change.
    select.replaceChildren();

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = columns.length
        ? "Select target column..."
        : "Upload dataset first...";
    select.appendChild(placeholder);

    columns.forEach((column) => {
        const option = document.createElement("option");
        option.value = column;
        option.textContent = column;
        select.appendChild(option);
    });

    select.disabled = columns.length === 0;
    select.setCustomValidity("");

    // Clear feedback from the previously selected target.
    if (targetFeedback) {
        targetFeedback.textContent = columns.length
            ? "Select the column you want to predict."
            : "Upload a dataset and select the column you want to predict.";

        targetFeedback.classList.remove("text-success", "text-danger");
    }

    // A new dataset requires a fresh target selection.
    if (trainButton) {
        trainButton.disabled = true;
    }
}

// Update dashboard metadata after an upload or replacement.
function updateDatasetDisplay(dataset) {
    const datasetMeta = document.getElementById("dataset-meta");

    if (datasetMeta) {
        datasetMeta.textContent =
            `${dataset.filename} Â· ${dataset.row_count} rows Â· ${dataset.column_count} columns`;
    }

    if (Array.isArray(dataset.columns)) {
        updateTargetColumnOptions(dataset.columns);
    }

    updatePreprocessingControls(true);
    updateTrainingControls(true);
}

// US-08 / US-11: Upload a new dataset or replace the active dataset.
async function uploadDataset(event) {
    event.preventDefault();

    const form = event.currentTarget;
    const fileInput = document.getElementById("csv-file");
    const button = document.getElementById("upload-btn");

    if (!fileInput || !fileInput.files.length) {
        showUploadFeedback("Choose a CSV file to upload.", false);
        return;
    }

    const isReplacement = Boolean(activeDatasetId);

    if (button) {
        button.disabled = true;
        button.textContent = isReplacement
            ? "Replacingâ€¦"
            : "Uploadingâ€¦";
    }

    const feedback = document.getElementById("upload-feedback");

    if (feedback) {
        feedback.classList.remove(
            "d-none",
            "alert-success",
            "alert-danger"
        );
        feedback.classList.add("alert-info");
        feedback.textContent = isReplacement
            ? "Validating and replacing the active datasetâ€¦"
            : "Validating and uploading your CSVâ€¦";
    }

    try {
        const endpoint = isReplacement
            ? `/dataset/${encodeURIComponent(activeDatasetId)}/replace`
            : "/dataset/upload";

        const result = await apiRequest(endpoint, {
            method: "POST",
            body: new FormData(form)
        });

        // Save the active dataset ID for this browser session.
        activeDatasetId = String(result.dataset_id);
        sessionStorage.setItem("activeDatasetId", activeDatasetId);

        updateDatasetDisplay(result);

        showUploadFeedback(
            isReplacement
                ? `${result.filename} replaced the active dataset successfully: ${result.row_count} rows and ${result.column_count} columns.`
                : `${result.filename} uploaded successfully: ${result.row_count} rows and ${result.column_count} columns.`,
            true
        );

        await loadDatasetPreview(result.dataset_id);
        await loadDatasetStatistics(result.dataset_id);

        form.reset();

    } catch (error) {
        showUploadFeedback(
            error.message ||
            "Dataset upload or replacement failed. Please try again.",
            false
        );
    } finally {
        if (button) {
            button.disabled = false;
            updateUploadButton();
        }
    }
}

function updateSplitDisplay() {
    const slider = document.getElementById("split-range");
    const display = document.getElementById("split-display");
    if (!slider || !display) return;
    const testPercent = Math.round(Number(slider.value) * 100);
    display.textContent = `${100 - testPercent}% Train / ${testPercent}% Test`;
}

async function evaluateDataset(event) {
    event.preventDefault();
    const targetColumn = document.getElementById("target-column-select").value;
    const algorithmSelect = document.getElementById("algorithm-select");
    const algorithm = algorithmSelect.value;
    const selectedOption = algorithmSelect.selectedOptions[0];
    const inferredTask = selectedOption?.parentElement?.label?.toLowerCase() === "regression"
        ? "regression"
        : "classification";
    const taskType = document.getElementById("task-type-select")?.value || inferredTask;
    const feedback = document.getElementById("evaluation-container");
    const trainButton = document.getElementById("train-btn");

    if (!activeDatasetId) {
        renderEvaluationMessage("Upload a dataset before training a model.", true);
        return;
    }
    if (!targetColumn || !algorithm) {
        renderEvaluationMessage("Choose a target column and algorithm.", true);
        return;
    }

    trainButton.disabled = true;
    feedback.textContent = "Training and evaluating the modelâ€¦";
    feedback.className = "card-body text-muted";
    try {
        const result = await apiRequest("/ml/evaluate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset_id: Number(activeDatasetId),
                target_column: targetColumn,
                algorithm,
                task_type: taskType,
                test_size: Number(document.getElementById("split-range").value)
            })
        });
        renderEvaluationResult(result);
    } catch (error) {
        renderEvaluationMessage(error.message || "The model could not be evaluated.", true);
    } finally {
        trainButton.disabled = false;
    }
}

function renderEvaluationMessage(message, isError = false) {
    const container = document.getElementById("evaluation-container");
    container.replaceChildren();
    const paragraph = document.createElement("p");
    paragraph.className = isError ? "mb-0 text-danger" : "mb-0 text-muted";
    paragraph.textContent = message;
    container.appendChild(paragraph);
}


function renderEvaluationResult(result) {
    const container = document.getElementById("evaluation-container");

    // Destroy the previous chart before replacing the results.
    const oldCanvas = container.querySelector("#evaluation-chart");
    if (oldCanvas && window.Chart) {
        const oldChart = Chart.getChart(oldCanvas);
        if (oldChart) oldChart.destroy();
    }

    container.replaceChildren();

    // Display train-test split information.
    const split = document.createElement("p");
    split.className = "small text-muted mb-3";
    split.textContent =
        `Training used ${result.split.train_rows} rows and testing used ${result.split.test_rows} rows.`;
    container.appendChild(split);

    const metrics = result.metrics || {};

    // Display evaluation metrics as cards.
    const metricsRow = document.createElement("div");
    metricsRow.className = "row g-3";

    const metricLabels = {
        accuracy: "Accuracy",
        precision_weighted: "Precision",
        recall_weighted: "Recall",
        f1_weighted: "F1 Score",
        rmse: "RMSE",
        r2: "R²",
        mse: "MSE"
    };

    for (const [name, value] of Object.entries(metrics)) {
        const column = document.createElement("div");
        column.className = "col-6 col-md-4";

        const card = document.createElement("div");
        card.className = "metric-card h-100";

        const label = document.createElement("div");
        label.className = "text-muted small mb-2";
        label.textContent = metricLabels[name] || name.replaceAll("_", " ");

        const metricValue = document.createElement("div");
        metricValue.className = "metric-value";
        metricValue.textContent =
            typeof value === "number" ? value.toFixed(4) : String(value);

        card.append(label, metricValue);
        column.appendChild(card);
        metricsRow.appendChild(column);
    }

    container.appendChild(metricsRow);

    // Create a chart for classification performance metrics.
    const classificationMetrics = [
        ["accuracy", "Accuracy"],
        ["precision_weighted", "Precision"],
        ["recall_weighted", "Recall"],
        ["f1_weighted", "F1 Score"]
    ];

    const availableMetrics = classificationMetrics.filter(
        ([key]) => typeof metrics[key] === "number"
    );

    if (availableMetrics.length > 0 && window.Chart) {
        const heading = document.createElement("h6");
        heading.className = "mt-4 mb-3";
        heading.textContent = "Classification Performance Chart";
        container.appendChild(heading);

        const chartContainer = document.createElement("div");
        chartContainer.style.position = "relative";
        chartContainer.style.height = "280px";

        const canvas = document.createElement("canvas");
        canvas.id = "evaluation-chart";
        canvas.setAttribute("aria-label", "Classification metrics bar chart");

        chartContainer.appendChild(canvas);
        container.appendChild(chartContainer);

        new Chart(canvas, {
            type: "bar",
            data: {
                labels: availableMetrics.map(([, label]) => label),
                datasets: [{
                    label: "Performance Score",
                    data: availableMetrics.map(([key]) => metrics[key]),
                    backgroundColor: [
                        "rgba(13, 110, 253, 0.75)",
                        "rgba(25, 135, 84, 0.75)",
                        "rgba(255, 193, 7, 0.75)",
                        "rgba(13, 202, 240, 0.75)"
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: "Score"
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    } else if (!window.Chart && availableMetrics.length > 0) {
        const warning = document.createElement("p");
        warning.className = "text-warning small mt-3";
        warning.textContent = "Chart.js could not be loaded. Metrics are still displayed above.";
        container.appendChild(warning);
    }

    // Display confusion matrix for classification.
    if (Array.isArray(result.confusion_matrix) && result.confusion_matrix.length) {
        const heading = document.createElement("h6");
        heading.className = "mt-4 mb-3";
        heading.textContent = "Confusion Matrix";
        container.appendChild(heading);

        const table = document.createElement("table");
        table.className =
            "table table-bordered table-sm confusion-matrix-table";

        result.confusion_matrix.forEach((rowValues) => {
            const row = document.createElement("tr");

            rowValues.forEach((value) => {
                const cell = document.createElement("td");
                cell.textContent = String(value);
                row.appendChild(cell);
            });

            table.appendChild(row);
        });

        container.appendChild(table);
    }

    // Refresh saved evaluation history.
    if (result.evaluation_id) {
        loadRunHistory();
    }
}

async function loadDatasetPreview(datasetId) {
    const status = document.getElementById("preview-status");
    const table = document.getElementById("preview-table");
    const header = table?.querySelector("thead");
    const body = table?.querySelector("tbody");
    const sampleLabel = document.getElementById("preview-sample-label");
    if (!status || !table || !header || !body) return;

    header.replaceChildren();
    body.replaceChildren();
    sampleLabel?.classList.add("d-none");
    status.className = "px-3 py-3 text-muted small";
    status.textContent = "Loading dataset sampleâ€¦";

    try {
        const preview = await apiRequest(`/dataset/${encodeURIComponent(datasetId)}/preview`);
        document.getElementById("dataset-meta").textContent =
            `${preview.row_count} rows Â· ${preview.column_count} columns`;

        const headerRow = document.createElement("tr");
        for (const column of preview.columns) {
            const cell = document.createElement("th");
            cell.scope = "col";
            cell.className = "p-3";
            const name = document.createElement("div");
            name.textContent = column;
            const type = document.createElement("span");
            type.className = "d-block fw-normal text-muted mt-1";
            type.textContent = preview.data_types[column] || "unknown";
            cell.append(name, type);
            headerRow.appendChild(cell);
        }
        header.appendChild(headerRow);

        for (const row of preview.rows) {
            const tableRow = document.createElement("tr");
            for (const column of preview.columns) {
                const cell = document.createElement("td");
                const value = row[column];
                cell.textContent = value === null || value === undefined ? "â€”" : String(value);
                tableRow.appendChild(cell);
            }
            body.appendChild(tableRow);
        }

        if (preview.sample_size === 0) {
            status.textContent = "This dataset has no data rows to preview.";
            status.className = "px-3 py-3 text-muted small";
        } else {
            status.textContent = `Showing ${preview.sample_size} sample row${preview.sample_size === 1 ? "" : "s"} of ${preview.row_count} total.`;
            status.className = "px-3 py-2 text-muted small";
        }
        sampleLabel?.classList.remove("d-none");
    } catch (error) {
        document.getElementById("dataset-meta").textContent = "Preview unavailable";
        status.className = "px-3 py-3 text-danger small";
        status.textContent = error.message || "Could not load the dataset preview.";
    }
}

// US-13: Load and display dataset statistics.
async function loadDatasetStatistics(datasetId) {
    const status = document.getElementById("statistics-status");
    const summary = document.getElementById("statistics-summary");
    const rowCount = document.getElementById("statistics-row-count");
    const columnCount = document.getElementById("statistics-column-count");
    const table = document.getElementById("statistics-table");
    const tableBody = document.getElementById("statistics-table-body");

    if (!status || !summary || !rowCount || !columnCount || !table || !tableBody) {
        return;
    }

    status.className = "text-muted small";
    status.textContent = "Loading dataset statistics...";
    summary.classList.add("d-none");
    table.classList.add("d-none");
    tableBody.replaceChildren();

    try {
        const data = await apiRequest(
            `/dataset/${encodeURIComponent(datasetId)}/statistics`
        );

        rowCount.textContent = data.row_count;
        columnCount.textContent = data.column_count;

        for (const column of data.columns) {
            const row = document.createElement("tr");

            const values = [
                column.name,
                column.type,
                column.missing_values,
                column.statistics?.mean,
                column.statistics?.median,
                column.statistics?.min,
                column.statistics?.max
            ];

            for (const value of values) {
                const cell = document.createElement("td");

                cell.textContent =
                    value === null || value === undefined
                        ? "â€”"
                        : typeof value === "number"
                            ? Number(value.toFixed(2)).toString()
                            : String(value);

                row.appendChild(cell);
            }

            tableBody.appendChild(row);
        }

        summary.classList.remove("d-none");
        table.classList.remove("d-none");
        status.textContent = `Statistics for ${data.filename}`;
    } catch (error) {
        status.className = "text-danger small";
        status.textContent =
            error.message || "Could not load dataset statistics.";
    }
}

// Display upload or replacement feedback.
function showUploadFeedback(message, success) {
    const feedback = document.getElementById("upload-feedback");
    if (!feedback) return;

    feedback.classList.remove(
        "d-none",
        "alert-info",
        "alert-success",
        "alert-danger"
    );

    feedback.classList.add(
        success ? "alert-success" : "alert-danger"
    );

    feedback.textContent = message;
}

// Restore active dataset metadata after a dashboard refresh.
async function restoreActiveDataset() {
    if (!activeDatasetId) {
        updateUploadButton();
        return;
    }

    try {
        const result = await apiRequest(
            `/dataset/${encodeURIComponent(activeDatasetId)}`
        );

        const dataset = result.dataset;

        updateDatasetDisplay({
            dataset_id: dataset.id,
            filename: dataset.filename,
            row_count: dataset.row_count,
            column_count: dataset.column_count,
            columns: dataset.metadata?.columns || []
        });

        await loadDatasetPreview(dataset.id);
        await loadDatasetStatistics(dataset.id);

    } catch (error) {
        // Clear stale IDs, including IDs belonging to another account.
        console.warn("Unable to restore active dataset:", error.message);

        activeDatasetId = null;
        sessionStorage.removeItem("activeDatasetId");

        const datasetMeta = document.getElementById("dataset-meta");
        if (datasetMeta) {
            datasetMeta.textContent = "No dataset selected";
        }

        updateTargetColumnOptions([]);
        updatePreprocessingControls(false);
        updateTrainingControls(false);
    }

    updateUploadButton();
}

// US-06: Handle logout.
async function handleLogout() {
    const logoutBtn = document.getElementById("logout-btn");
    const feedback = document.getElementById("logout-feedback");

    if (logoutBtn) {
        logoutBtn.disabled = true;
        logoutBtn.textContent = "Logging out...";
    }

    if (feedback) {
        feedback.classList.add("d-none");
        feedback.textContent = "";
    }

    try {
        const result = await apiRequest("/auth/logout", {
            method: "POST"
        });

        // Do not reuse one user's active dataset in another session.
        activeDatasetId = null;
        sessionStorage.removeItem("activeDatasetId");

        sessionStorage.setItem(
            "logoutMessage",
            result.message || "You have been logged out successfully"
        );

        sessionStorage.setItem("logoutStatus", "success");

        window.location.assign(result.redirect_url || "/login");

    } catch (error) {
        console.error("Logout failed:", error.message);

        if (feedback) {
            feedback.textContent =
                error.message || "Logout failed. Please try again.";
            feedback.className = "alert alert-danger";
        }

        if (logoutBtn) {
            logoutBtn.disabled = false;
            logoutBtn.textContent = "Logout";
        }
    }
}

// Login form.
function setupLoginForm() {
    const logoutMessage = sessionStorage.getItem("logoutMessage");
    const logoutStatus = sessionStorage.getItem("logoutStatus");

    if (logoutMessage) {
        const loginFeedback = document.getElementById("login-feedback");

        if (loginFeedback) {
            loginFeedback.textContent = logoutMessage;
            loginFeedback.className =
                logoutStatus === "error"
                    ? "mt-3 alert alert-danger"
                    : "mt-3 alert alert-success";
        }

        sessionStorage.removeItem("logoutMessage");
        sessionStorage.removeItem("logoutStatus");
    }

    document.getElementById("login-form")
        ?.addEventListener("submit", async (event) => {
            event.preventDefault();

            const feedback = document.getElementById("login-feedback");
            const submitButton = event.currentTarget.querySelector(
                'button[type="submit"]'
            );

            const credentials = {
                username: document.getElementById("login-username").value.trim(),
                password: document.getElementById("login-password").value
            };

            if (submitButton) {
                submitButton.disabled = true;
            }

            if (feedback) {
                feedback.textContent = "Signing in...";
                feedback.className = "mt-3 text-muted";
            }

            try {
                const result = await apiRequest("/auth/login", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(credentials)
                });

                window.location.assign(result.redirect_url || "/");

            } catch (error) {
                if (feedback) {
                    feedback.textContent = error.message;
                    feedback.className = "mt-3 alert alert-danger";
                }
            } finally {
                if (submitButton) {
                    submitButton.disabled = false;
                }
            }
        });
}

// Restore authenticated session and active dataset.
async function restoreSession() {
    const currentUser = document.getElementById("current-user");
    const logoutBtn = document.getElementById("logout-btn");

    if (!currentUser) return;

    try {
        const result = await apiRequest("/auth/session");

        if (!result.authenticated) {
            activeDatasetId = null;
            sessionStorage.removeItem("activeDatasetId");
            window.location.replace("/login");
            return;
        }

        currentUser.textContent = result.user.username;

        const roleBadge = document.getElementById("user-role-badge");
        const adminBtn = document.getElementById("admin-panel-btn");
        const userRole = result.user.role || "student";

        if (roleBadge) {
            roleBadge.textContent = userRole;
            roleBadge.classList.remove("d-none");
        }

        if (adminBtn) {
            if (userRole === "admin") {
                adminBtn.classList.remove("d-none");
            } else {
                adminBtn.classList.add("d-none");
            }
        }

        if (logoutBtn) {
            logoutBtn.classList.remove("d-none");
        }

        const readOnly = userRole === "instructor";
        document.getElementById("instructor-mode-banner")?.classList.toggle("d-none", !readOnly);
        for (const id of ["upload-form", "preprocess-form", "train-form"]) {
            const form = document.getElementById(id);
            form?.closest(".card")?.classList.toggle("d-none", readOnly);
        }
        for (const id of ["dataset-preview-card", "statistics-card", "evaluation-results-card"]) {
            document.getElementById(id)?.classList.toggle("d-none", readOnly);
        }
        document.getElementById("run-history-title").textContent = readOnly
            ? "Shared Evaluation Review" : "My Run History";
        if (readOnly) {
            activeDatasetId = null;
            sessionStorage.removeItem("activeDatasetId");
        } else {
            await loadAlgorithmCatalog();
            await restoreActiveDataset();
        }
        await loadRunHistory();

    } catch {
        activeDatasetId = null;
        sessionStorage.removeItem("activeDatasetId");
        window.location.replace("/login");
    }
}

// Registration form (US-04).
function setupRegistrationForm() {
    document.getElementById("registration-form")
        ?.addEventListener("submit", async (event) => {
            event.preventDefault();

            const feedback = document.getElementById(
                "registration-feedback"
            );

            const data = {
                username: document.getElementById("reg-username").value.trim(),
                email: document.getElementById("reg-email").value.trim(),
                password: document.getElementById("reg-password").value
            };

            try {
                const response = await apiRequest("/auth/register", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                });

                if (feedback) {
                    feedback.textContent =
                        response.message || "Registration successful";
                    feedback.className = "mt-3 alert alert-success";
                }

                event.target.reset();

            } catch (error) {
                if (feedback) {
                    feedback.textContent =
                        error.message || "Registration failed";
                    feedback.className = "mt-3 alert alert-danger";
                }

                console.error("Registration failed:", error.message);
            }
        });
}

// â”€â”€â”€ Administrative Control Center (US-14a / FR-7.1, 7.2, 7.3) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function setupAdminPanel() {
    const adminModal = document.getElementById("adminModal");
    if (!adminModal) return;

    adminModal.addEventListener("show.bs.modal", () => {
        loadAdminUsers();
        loadAdminDatasets();
        loadAdminLogs();
    });

    document.getElementById("refresh-admin-users-btn")?.addEventListener("click", loadAdminUsers);
    document.getElementById("refresh-admin-datasets-btn")?.addEventListener("click", loadAdminDatasets);
    document.getElementById("refresh-admin-logs-btn")?.addEventListener("click", loadAdminLogs);

    document.getElementById("tab-users")?.addEventListener("click", loadAdminUsers);
    document.getElementById("tab-datasets")?.addEventListener("click", loadAdminDatasets);
    document.getElementById("tab-logs")?.addEventListener("click", loadAdminLogs);
}

async function loadAdminUsers() {
    const tbody = document.getElementById("admin-users-table-body");
    if (!tbody) return;

    try {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted py-3">Loading users...</td></tr>';
        const data = await apiRequest("/admin/users");
        if (!data.users || data.users.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted py-3">No registered users found.</td></tr>';
            return;
        }

        tbody.innerHTML = "";
        data.users.forEach(u => {
            const tr = document.createElement("tr");
            const statusBadge = u.is_active
                ? '<span class="badge bg-success">Active</span>'
                : '<span class="badge bg-danger">Deactivated</span>';
            const actionBtn = u.role === "admin"
                ? '<span class="text-muted small">Protected</span>'
                : `<button class="btn btn-sm ${u.is_active ? 'btn-outline-danger' : 'btn-outline-success'}" onclick="toggleUserStatus(${u.id}, ${u.is_active})">${u.is_active ? 'Deactivate' : 'Activate'}</button>`;

            tr.innerHTML = `
                <td>${u.id}</td>
                <td class="fw-semibold">${u.username}</td>
                <td>${u.email}</td>
                <td><span class="badge bg-secondary text-uppercase">${u.role}</span></td>
                <td>${statusBadge}</td>
                <td class="text-muted small">${u.created_at ? new Date(u.created_at).toLocaleDateString() : '-'}</td>
                <td>${actionBtn}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (error) {
        tbody.innerHTML = `<tr><td colspan="7" class="text-center text-danger py-3">Failed to load users: ${error.message}</td></tr>`;
    }
}

async function toggleUserStatus(userId, currentStatus) {
    if (!confirm(`Are you sure you want to ${currentStatus ? 'deactivate' : 'activate'} this user account?`)) return;

    try {
        await apiRequest(`/admin/users/${userId}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ is_active: !currentStatus })
        });
        await loadAdminUsers();
        await loadAdminLogs();
    } catch (error) {
        alert(`Error updating user status: ${error.message}`);
    }
}

async function loadAdminDatasets() {
    const tbody = document.getElementById("admin-datasets-table-body");
    if (!tbody) return;

    try {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted py-3">Loading datasets...</td></tr>';
        const data = await apiRequest("/admin/datasets");
        if (!data.datasets || data.datasets.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted py-3">No stored datasets found.</td></tr>';
            return;
        }

        tbody.innerHTML = "";
        data.datasets.forEach(d => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${d.id}</td>
                <td class="fw-semibold"><i class="bi bi-person me-1 text-primary"></i>${d.owner_username} (ID: ${d.user_id})</td>
                <td>${d.filename}</td>
                <td>${d.row_count}</td>
                <td>${d.column_count}</td>
                <td class="text-muted small">${d.created_at ? new Date(d.created_at).toLocaleString() : '-'}</td>
                <td>
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteAdminDataset(${d.id})">
                        <i class="bi bi-trash me-1"></i>Delete
                    </button>
                </td>
            `;
            tbody.appendChild(tr);
        });
    } catch (error) {
        tbody.innerHTML = `<tr><td colspan="7" class="text-center text-danger py-3">Failed to load datasets: ${error.message}</td></tr>`;
    }
}

async function deleteAdminDataset(datasetId) {
    if (!confirm(`Are you sure you want to delete dataset #${datasetId}? This will remove the file permanently.`)) return;

    try {
        await apiRequest(`/admin/datasets/${datasetId}`, {
            method: "DELETE"
        });
        await loadAdminDatasets();
        await loadAdminLogs();
    } catch (error) {
        alert(`Error deleting dataset: ${error.message}`);
    }
}

async function loadAdminLogs() {
    const tbody = document.getElementById("admin-logs-table-body");
    if (!tbody) return;

    try {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted py-3">Loading activity logs...</td></tr>';
        const data = await apiRequest("/admin/logs");
        if (!data.logs || data.logs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted py-3">No activity logs recorded yet.</td></tr>';
            return;
        }

        tbody.innerHTML = "";
        data.logs.forEach(l => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td class="text-nowrap small text-muted">${l.timestamp ? new Date(l.timestamp).toLocaleString() : '-'}</td>
                <td><span class="badge bg-primary">${l.action}</span></td>
                <td>${l.username ? `${l.username} (ID: ${l.user_id})` : '<span class="text-muted">system</span>'}</td>
                <td class="small text-break">${l.details || '-'}</td>
                <td class="small text-muted">${l.ip_address || '-'}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (error) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-center text-danger py-3">Failed to load logs: ${error.message}</td></tr>`;
    }
}


async function loadAlgorithmCatalog() {
    const select = document.getElementById("algorithm-select");
    if (!select) return;
    try {
        const catalog = await apiRequest("/ml/algorithms");
        select.replaceChildren(new Option("Choose an algorithm...", ""));
        for (const [task, algorithms] of Object.entries(catalog)) {
            const group = document.createElement("optgroup");
            group.label = task === "regression" ? "Regression" : "Classification";
            for (const item of algorithms) group.appendChild(new Option(item.label, item.id));
            select.appendChild(group);
        }
    } catch (error) { console.error("Algorithm catalog unavailable:", error.message); }
}

async function loadRunHistory() {
    const body = document.getElementById("run-history-body");
    if (!body) return;
    try {
        const path = document.getElementById("user-role-badge")?.textContent === "instructor"
            ? "/ml/instructor/evaluations" : "/ml/evaluations";
        const data = await apiRequest(path);
        body.replaceChildren();
        const readOnly = path.includes("instructor");
        const header = document.getElementById("run-history-header");
        if (header) {
            const labels = readOnly
                ? ["Select", "Run", "Model", "Metrics", "Student", "Review"]
                : ["Select", "Run", "Model", "Metrics", "Shared", "Downloads"];
            header.replaceChildren(...labels.map((label) => {
                const cell = document.createElement("th");
                cell.textContent = label;
                return cell;
            }));
        }
        document.getElementById("instructor-review-details")?.classList.toggle("d-none", !readOnly);
        for (const run of data.evaluations || []) {
            const row = document.createElement("tr");
            const metrics = Object.entries(run.metrics || {}).map(([key, value]) => `${key}: ${typeof value === "number" ? value.toFixed(3) : value}`).join(" · ");
            row.innerHTML = `<td><input class="form-check-input compare-run" type="checkbox" value="${run.id}" aria-label="Select run ${run.id}"></td><td>${run.id}</td><td></td><td></td><td></td><td></td>`;
            row.cells[2].textContent = run.model_name || run.algorithm;
            row.cells[3].textContent = metrics;
            row.cells[4].textContent = readOnly ? (run.owner || "Shared") : (run.shared_with_instructors ? "Shared" : "Private");
            const actions = row.cells[5];
            if (readOnly) {
                const view = document.createElement("button");
                view.type = "button";
                view.className = "btn btn-sm btn-outline-primary";
                view.textContent = "View results";
                view.addEventListener("click", () => loadInstructorEvaluation(run.id));
                actions.appendChild(view);
            } else {
                const share = document.createElement("button"); share.className = "btn btn-sm btn-outline-secondary me-1"; share.textContent = run.shared_with_instructors ? "Unshare" : "Share";
                share.onclick = async () => { await apiRequest(`/ml/evaluations/${run.id}/sharing`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ shared_with_instructors: !run.shared_with_instructors }) }); loadRunHistory(); };
                actions.appendChild(share);
                const modelId = run.model_id;
                const modelDownload = document.createElement("button");
                modelDownload.type = "button"; modelDownload.className = "btn btn-sm btn-outline-primary me-1";
                modelDownload.textContent = "Model";
                modelDownload.addEventListener("click", () => downloadFile(`/ml/models/${modelId}/download`, `model-${modelId}.joblib`));
                actions.appendChild(modelDownload);
                const reportDownload = document.createElement("button");
                reportDownload.type = "button"; reportDownload.className = "btn btn-sm btn-outline-primary";
                reportDownload.textContent = "PDF";
                reportDownload.addEventListener("click", () => downloadFile(`/ml/evaluations/${run.id}/report`, `evaluation-${run.id}.pdf`));
                actions.appendChild(reportDownload);
            }
            body.appendChild(row);
        }
        if (!body.children.length) body.innerHTML = '<tr><td colspan="6">No shared evaluations are available.</td></tr>';
        document.getElementById("compare-runs-btn")?.classList.remove("d-none");
    } catch (error) {
        body.replaceChildren();
        const row = document.createElement("tr"); const cell = document.createElement("td");
        cell.colSpan = 6; cell.textContent = error.message; row.appendChild(cell); body.appendChild(row);
    }
}

async function downloadFile(endpoint, filename) {
    const target = document.getElementById("comparison-results");
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, { credentials: "same-origin" });
        if (!response.ok) {
            const contentType = response.headers.get("content-type") || "";
            const data = contentType.includes("application/json") ? await response.json() : {};
            throw new Error(data.error || `Download failed (HTTP ${response.status}).`);
        }
        const url = URL.createObjectURL(await response.blob());
        const link = document.createElement("a");
        link.href = url; link.download = filename; document.body.appendChild(link); link.click(); link.remove();
        URL.revokeObjectURL(url);
        if (target) target.textContent = `${filename} downloaded.`;
    } catch (error) {
        if (target) target.textContent = error.message || `Could not download ${filename}.`;
    }
}

async function loadInstructorEvaluation(evaluationId) {
    const panel = document.getElementById("instructor-review-details");
    if (!panel) return;
    panel.classList.remove("d-none");
    panel.replaceChildren();
    const status = document.createElement("p");
    status.className = "text-muted";
    status.textContent = "Loading shared evaluation…";
    panel.appendChild(status);
    try {
        const { evaluation } = await apiRequest(`/ml/instructor/evaluations/${evaluationId}`);
        panel.replaceChildren();
        const title = document.createElement("h5");
        title.textContent = `Run ${evaluation.id}: ${evaluation.model_name} · ${evaluation.algorithm}`;
        panel.appendChild(title);
        const owner = document.createElement("p");
        owner.className = "text-muted small";
        owner.textContent = `Shared by ${evaluation.owner} · ${evaluation.created_at || "Date unavailable"}`;
        panel.appendChild(owner);

        const metricHeading = document.createElement("h6");
        metricHeading.textContent = "Performance metrics";
        panel.appendChild(metricHeading);
        const metricsTable = document.createElement("table");
        metricsTable.className = "table table-sm table-bordered w-auto";
        const metricsBody = document.createElement("tbody");
        for (const [name, value] of Object.entries(evaluation.metrics || {})) {
            const row = document.createElement("tr");
            const label = document.createElement("th");
            label.scope = "row";
            label.textContent = name;
            const result = document.createElement("td");
            result.textContent = typeof value === "number" ? value.toFixed(4) : String(value ?? "—");
            row.append(label, result);
            metricsBody.appendChild(row);
        }
        metricsTable.appendChild(metricsBody);
        panel.appendChild(metricsTable);

        if (Array.isArray(evaluation.confusion_matrix)) {
            const heading = document.createElement("h6");
            heading.textContent = "Confusion matrix";
            panel.appendChild(heading);
            const matrixTable = document.createElement("table");
            matrixTable.className = "table table-sm table-bordered w-auto";
            const matrixHead = document.createElement("thead");
            const headerRow = document.createElement("tr");
            const corner = document.createElement("th");
            corner.textContent = "Actual / predicted";
            headerRow.appendChild(corner);
            const columnCount = evaluation.confusion_matrix[0]?.length || 0;
            for (let column = 0; column < columnCount; column += 1) {
                const header = document.createElement("th");
                header.scope = "col";
                const classLabel = evaluation.confusion_matrix_labels?.[column];
                header.textContent = `Predicted ${classLabel ?? column}`;
                headerRow.appendChild(header);
            }
            matrixHead.appendChild(headerRow);
            matrixTable.appendChild(matrixHead);
            const matrixBody = document.createElement("tbody");
            evaluation.confusion_matrix.forEach((values, index) => {
                const row = document.createElement("tr");
                const label = document.createElement("th");
                label.scope = "row";
                const classLabel = evaluation.confusion_matrix_labels?.[index];
                label.textContent = `Actual ${classLabel ?? index}`;
                row.appendChild(label);
                values.forEach((value) => {
                    const cell = document.createElement("td");
                    cell.textContent = String(value);
                    row.appendChild(cell);
                });
                matrixBody.appendChild(row);
            });
            matrixTable.appendChild(matrixBody);
            panel.appendChild(matrixTable);
        }
        if (evaluation.report && typeof evaluation.report === "object") {
            const reportHeading = document.createElement("h6");
            reportHeading.textContent = "Classification report";
            panel.appendChild(reportHeading);
            const report = document.createElement("pre");
            report.className = "small bg-light border rounded p-2";
            report.textContent = JSON.stringify(evaluation.report, null, 2);
            panel.appendChild(report);
        }
    } catch (error) {
        panel.replaceChildren();
        const message = document.createElement("p");
        message.className = "alert alert-danger";
        message.textContent = error.message;
        panel.appendChild(message);
    }
}

async function compareSelectedRuns() {
    const selected = Array.from(document.querySelectorAll(".compare-run:checked"), item => Number(item.value));
    const target = document.getElementById("comparison-results");
    if (selected.length < 2) { target.textContent = "Select at least two runs to compare."; return; }
    try {
        const instructor = document.getElementById("user-role-badge")?.textContent === "instructor";
        const options = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ evaluation_ids: selected }) };
        const comparison = instructor
            ? await apiRequest("/ml/instructor/compare", options)
            : await apiRequest("/ml/compare", options);
        const evaluations = comparison.evaluations || [];
        target.replaceChildren();
        const table = document.createElement("table"); table.className = "table table-bordered table-sm";
        const head = document.createElement("tr"); head.innerHTML = "<th>Metric</th>";
        const modelRows = comparison.models || [];
        const labels = modelRows.length
            ? modelRows.map((model) => model.run_label || model.model_name)
            : evaluations.map((run) => `Run ${run.id} · ${run.algorithm}`);
        labels.forEach((label) => { const th = document.createElement("th"); th.textContent = label; head.appendChild(th); });
        const thead = document.createElement("thead"); thead.appendChild(head); table.appendChild(thead);
        const metrics = comparison.metric_names || [...new Set(evaluations.flatMap(run => Object.keys(run.metrics || {})))];
        const tbody = document.createElement("tbody");
        metrics.forEach(metric => { const row = document.createElement("tr"); const name = document.createElement("th"); name.textContent = metric; row.appendChild(name); evaluations.forEach(run => { const td = document.createElement("td"); td.textContent = run.metrics?.[metric] ?? "—"; row.appendChild(td); }); tbody.appendChild(row); });
        table.appendChild(tbody); target.appendChild(table);

        if (comparison.chart?.metrics && modelRows.length) {
            const title = document.createElement("h6");
            title.className = "mt-3";
            title.textContent = `Metric comparison chart · ${comparison.task_type}`;
            target.appendChild(title);
            const palette = ["#0d6efd", "#198754", "#fd7e14", "#6f42c1", "#dc3545", "#20c997"];
            const rowHeight = 22;
            const metricHeight = 26 + modelRows.length * rowHeight;
            const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            svg.setAttribute("viewBox", `0 0 680 ${metrics.length * metricHeight + 8}`);
            svg.setAttribute("role", "img");
            svg.setAttribute("aria-label", `Comparison chart for ${comparison.task_type} metrics`);
            svg.classList.add("w-100");
            metrics.forEach((metric, metricIndex) => {
                const y = metricIndex * metricHeight + 18;
                const metricLabel = document.createElementNS(svg.namespaceURI, "text");
                metricLabel.setAttribute("x", "0"); metricLabel.setAttribute("y", String(y));
                metricLabel.textContent = metric.toUpperCase(); svg.appendChild(metricLabel);
                const points = comparison.chart.metrics[metric] || [];
                const values = points.map((point) => Number(point.value));
                const max = Math.max(0.000001, ...values.map(Math.abs));
                const baseline = 300;
                const axis = document.createElementNS(svg.namespaceURI, "line");
                axis.setAttribute("x1", String(baseline)); axis.setAttribute("x2", String(baseline));
                axis.setAttribute("y1", String(y + 2)); axis.setAttribute("y2", String(y + modelRows.length * rowHeight + 4));
                axis.setAttribute("stroke", "#6c757d"); svg.appendChild(axis);
                points.forEach((point, index) => {
                    const model = modelRows[index];
                    const value = Number(point.value); const width = Math.max(1, Math.abs(value) / max * 170);
                    const bar = document.createElementNS(svg.namespaceURI, "rect");
                    bar.setAttribute("x", String(value < 0 ? baseline - width : baseline));
                    bar.setAttribute("y", String(y + 7 + index * rowHeight));
                    bar.setAttribute("width", String(width)); bar.setAttribute("height", "13");
                    bar.setAttribute("fill", palette[index % palette.length]); svg.appendChild(bar);
                    const valueLabel = document.createElementNS(svg.namespaceURI, "text");
                    valueLabel.setAttribute("x", String(value < 0 ? Math.max(0, baseline - width - 48) : baseline + width + 5));
                    valueLabel.setAttribute("y", String(y + 18 + index * rowHeight));
                    valueLabel.textContent = value.toPrecision(4); svg.appendChild(valueLabel);
                    const runLabel = document.createElementNS(svg.namespaceURI, "text");
                    runLabel.setAttribute("x", "485"); runLabel.setAttribute("y", String(y + 18 + index * rowHeight));
                    runLabel.textContent = `${model.run_label || model.model_name}`;
                    svg.appendChild(runLabel);
                });
            });
            target.appendChild(svg);
            const limitation = document.createElement("p");
            limitation.className = "small text-muted";
            limitation.textContent = comparison.limitations || "Each value comes from the run's persisted evaluation.";
            target.appendChild(limitation);
        }
    } catch (error) { target.textContent = error.message; }
}
