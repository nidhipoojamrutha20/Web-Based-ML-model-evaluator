
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

    // US-14: Data Preprocessing form listener.
    document.getElementById("preprocess-form")
        ?.addEventListener("submit", applyPreprocessing);

    // US-15: Validate and track the selected target column.
    const targetSelect = document.getElementById("target-column-select");
    const targetFeedback = document.getElementById("target-column-feedback");
    const trainButton = document.getElementById("train-btn");

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
            if (trainButton) updateTrainAvailability();
        });
    }
    // TODO: US-16 - Split ratio range slider listener (Nidhi).
    document.getElementById("train-form")?.addEventListener("submit", trainAndEvaluateModel);
    document.getElementById("compare-models-btn")?.addEventListener("click", compareSelectedModels);
    document.getElementById("task-type-select")?.addEventListener("change", updateTrainAvailability);
    document.getElementById("algorithm-select")?.addEventListener("change", updateTrainAvailability);
    document.getElementById("split-range")?.addEventListener("input", (event) => {
        const testPercent = Math.round(Number(event.currentTarget.value) * 100);
        const display = document.getElementById("split-display");
        if (display) display.textContent = `${100 - testPercent}% Train / ${testPercent}% Test`;
    });

    const logoutBtn = document.getElementById("logout-btn");

    if (logoutBtn) {
        logoutBtn.addEventListener("click", handleLogout);
    }

    console.log("Frontend architecture initialized.");
}

function updateTrainAvailability() {
    const button = document.getElementById("train-btn");
    const target = document.getElementById("target-column-select")?.value;
    const task = document.getElementById("task-type-select")?.value;
    const algorithm = document.getElementById("algorithm-select")?.value;
    if (button) button.disabled = !(activeDatasetId && target && task && algorithm);
}

async function trainAndEvaluateModel(event) {
    event.preventDefault();
    const button = document.getElementById("train-btn");
    const container = document.getElementById("evaluation-container");
    button.disabled = true;
    button.textContent = "Training…";
    if (container) container.textContent = "Training model and calculating held-out metrics…";
    try {
        const testSize = Number(document.getElementById("split-range")?.value || 0.2);
        const result = await apiRequest("/ml/train", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset_id: Number(activeDatasetId),
                target_column: document.getElementById("target-column-select")?.value,
                algorithm: document.getElementById("algorithm-select")?.value,
                task_type: document.getElementById("task-type-select")?.value,
                test_size: testSize
            })
        });
        renderEvaluation(result.evaluation, result.model);
        await loadComparisonRuns();
    } catch (error) {
        if (container) {
            container.textContent = error.message || "Training failed.";
            container.className = "card-body text-danger";
        }
    } finally {
        if (button) {
            button.innerHTML = '<i class="bi bi-play-circle me-1"></i>Train & Evaluate Model';
            updateTrainAvailability();
        }
    }
}

function renderEvaluation(evaluation, model) {
    const container = document.getElementById("evaluation-container");
    if (!container) return;
    container.replaceChildren();
    const heading = document.createElement("h6");
    heading.textContent = `${model.model_name} · ${model.dataset_name || "dataset"}`;
    container.appendChild(heading);
    const metrics = document.createElement("dl");
    metrics.className = "row small";
    Object.entries(evaluation.metrics || {}).forEach(([name, value]) => {
        const term = document.createElement("dt"); term.className = "col-6"; term.textContent = name.toUpperCase();
        const detail = document.createElement("dd"); detail.className = "col-6"; detail.textContent = Number(value).toPrecision(5);
        metrics.append(term, detail);
    });
    container.appendChild(metrics);
    if (evaluation.confusion_matrix) {
        const label = document.createElement("h6"); label.textContent = "Confusion matrix"; container.appendChild(label);
        const pre = document.createElement("pre"); pre.className = "small"; pre.textContent = JSON.stringify(evaluation.confusion_matrix); container.appendChild(pre);
    }
}

async function loadComparisonRuns() {
    const status = document.getElementById("comparison-status");
    const runs = document.getElementById("comparison-runs");
    if (!status || !runs) return;
    status.textContent = "Loading your trained runs…";
    runs.replaceChildren();
    try {
        const data = await apiRequest("/ml/models");
        const trained = (data.models || []).filter((model) => model.status === "trained");
        if (!trained.length) { status.textContent = "No trained runs yet. Train a model to get started."; updateCompareButton(); return; }
        status.textContent = "Choose runs for comparison or download their results.";
        trained.forEach((model) => {
            const col = document.createElement("div"); col.className = "col-12 col-md-6";
            const box = document.createElement("div"); box.className = "border rounded p-2 h-100";
            const label = document.createElement("label"); label.className = "form-check small";
            const check = document.createElement("input"); check.type = "checkbox"; check.className = "form-check-input comparison-run"; check.value = String(model.id);
            check.addEventListener("change", updateCompareButton);
            const text = document.createElement("span"); text.className = "form-check-label";
            text.textContent = `${model.run_label || `${model.model_name} · Run #${model.id}`} · ${model.dataset_name || `Dataset ${model.dataset_id}`}`;
            label.append(check, text); box.appendChild(label);
            const actions = document.createElement("div"); actions.className = "d-flex gap-2 mt-2";
            const modelDownload = document.createElement("button"); modelDownload.type = "button"; modelDownload.className = "btn btn-outline-secondary btn-sm"; modelDownload.textContent = "Download model";
            modelDownload.addEventListener("click", () => downloadFile(`/ml/models/${model.id}/download`, `model-${model.id}.joblib`));
            actions.appendChild(modelDownload);
            if (model.evaluation_id) {
                const reportDownload = document.createElement("button"); reportDownload.type = "button"; reportDownload.className = "btn btn-outline-secondary btn-sm"; reportDownload.textContent = "Download PDF report";
                reportDownload.addEventListener("click", () => downloadFile(`/ml/evaluations/${model.evaluation_id}/report`, `evaluation-${model.evaluation_id}.pdf`));
                actions.appendChild(reportDownload);
            }
            box.appendChild(actions); col.appendChild(box); runs.appendChild(col);
        });
        updateCompareButton();
    } catch (error) { status.textContent = error.message || "Could not load trained model runs."; status.className = "text-danger small"; }
}

function updateCompareButton() {
    const selected = document.querySelectorAll(".comparison-run:checked").length;
    const button = document.getElementById("compare-models-btn");
    if (button) button.disabled = selected < 2;
}

async function compareSelectedModels() {
    const selected = Array.from(document.querySelectorAll(".comparison-run:checked"), (input) => Number(input.value));
    const feedback = document.getElementById("comparison-feedback");
    const resultNode = document.getElementById("comparison-results");
    const button = document.getElementById("compare-models-btn");
    button.disabled = true; feedback.textContent = "Comparing saved evaluation results…";
    resultNode.replaceChildren();
    try {
        const comparison = await apiRequest("/ml/compare", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_ids: selected })
        });
        renderComparison(comparison, resultNode);
        feedback.textContent = `Compared ${comparison.models.length} ${comparison.task_type} runs using persisted evaluation metrics.`;
        feedback.className = "small mt-3 text-success";
    } catch (error) { feedback.textContent = error.message || "Comparison failed."; feedback.className = "small mt-3 text-danger"; }
    finally { updateCompareButton(); }
}

function renderComparison(data, root) {
    const table = document.createElement("table"); table.className = "table table-sm table-bordered align-middle";
    const head = document.createElement("thead"); const headingRow = document.createElement("tr");
    ["Metric", ...data.models.map((model) => model.run_label || model.model_name)].forEach((name) => { const th = document.createElement("th"); th.scope = "col"; th.textContent = name; headingRow.appendChild(th); });
    head.appendChild(headingRow); table.appendChild(head);
    const body = document.createElement("tbody");
    data.metric_names.forEach((metric) => {
        const row = document.createElement("tr"); const label = document.createElement("th"); label.scope = "row"; label.textContent = metric.toUpperCase(); row.appendChild(label);
        data.models.forEach((model) => { const cell = document.createElement("td"); const value = model.metrics[metric]; cell.textContent = typeof value === "number" ? value.toPrecision(5) : "Unavailable"; row.appendChild(cell); });
        body.appendChild(row);
    });
    table.appendChild(body); root.appendChild(table);
    const title = document.createElement("h6"); title.textContent = `Metric comparison chart · ${data.task_type}`; root.appendChild(title);
    const metricHeight = 35 + data.models.length * 18;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); svg.setAttribute("viewBox", `0 0 640 ${data.metric_names.length * metricHeight + 10}`); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", `Bar chart of ${data.task_type} metric values for ${data.models.map((m) => m.run_label || m.model_name).join(", ")}`); svg.classList.add("w-100");
    const palette = ["#0d6efd", "#198754", "#fd7e14", "#6f42c1", "#dc3545", "#20c997"];
    data.metric_names.forEach((metric, metricIndex) => {
        const y = metricIndex * metricHeight + 15; const label = document.createElementNS(svg.namespaceURI, "text"); label.setAttribute("x", "0"); label.setAttribute("y", String(y)); label.textContent = metric.toUpperCase(); svg.appendChild(label);
        const values = data.chart.metrics[metric].map((entry) => Number(entry.value)); const max = Math.max(0.000001, ...values.map(Math.abs));
        const zero = document.createElementNS(svg.namespaceURI, "line"); zero.setAttribute("x1", "300"); zero.setAttribute("x2", "300"); zero.setAttribute("y1", String(y + 2)); zero.setAttribute("y2", String(y + 10 + data.models.length * 18)); zero.setAttribute("stroke", "#6c757d"); zero.setAttribute("stroke-width", "1"); svg.appendChild(zero);
        values.forEach((value, modelIndex) => {
            const rowY = y + 8 + modelIndex * 18;
            const barWidth = Math.max(1, Math.abs(value) / max * 165);
            const bar = document.createElementNS(svg.namespaceURI, "rect"); bar.setAttribute("x", String(value < 0 ? 300 - barWidth : 300)); bar.setAttribute("y", String(rowY - 10)); bar.setAttribute("width", String(barWidth)); bar.setAttribute("height", "12"); bar.setAttribute("fill", palette[modelIndex % palette.length]); svg.appendChild(bar);
            const valueLabel = document.createElementNS(svg.namespaceURI, "text"); valueLabel.setAttribute("x", String(value < 0 ? Math.max(0, 295 - barWidth - 45) : 305 + barWidth)); valueLabel.setAttribute("y", String(rowY)); valueLabel.textContent = value.toPrecision(4); svg.appendChild(valueLabel);
        });
    });
    root.appendChild(svg);
    const legend = document.createElement("ul"); legend.className = "list-inline small";
    data.models.forEach((model, index) => { const item = document.createElement("li"); item.className = "list-inline-item me-3"; const swatch = document.createElement("span"); swatch.setAttribute("aria-hidden", "true"); swatch.style.display = "inline-block"; swatch.style.width = "0.8rem"; swatch.style.height = "0.8rem"; swatch.style.backgroundColor = palette[index % palette.length]; swatch.className = "me-1"; item.append(swatch, document.createTextNode(model.run_label || model.model_name)); legend.appendChild(item); });
    root.appendChild(legend);
    const limitation = document.createElement("p"); limitation.className = "small text-muted"; limitation.textContent = data.limitations; root.appendChild(limitation);
}

async function downloadFile(endpoint, filename) {
    const feedback = document.getElementById("comparison-feedback");
    feedback.textContent = `Preparing ${filename}…`;
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, { credentials: "same-origin" });
        if (!response.ok) {
            const type = response.headers.get("content-type") || "";
            const data = type.includes("application/json") ? await response.json() : {};
            throw new Error(data.error || `Download failed (HTTP ${response.status}).`);
        }
        const blob = await response.blob(); const url = URL.createObjectURL(blob);
        const link = document.createElement("a"); link.href = url; link.download = filename; document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url);
        feedback.textContent = `${filename} downloaded.`; feedback.className = "small mt-3 text-success";
    } catch (error) { feedback.textContent = error.message || `Could not download ${filename}.`; feedback.className = "small mt-3 text-danger"; }
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
        button.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>Applying Preprocessing…';
    }

    showPreprocessFeedback("Executing preprocessing transformations on dataset…", true);
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
            `${dataset.filename} · ${dataset.row_count} rows · ${dataset.column_count} columns`;
    }

    if (Array.isArray(dataset.columns)) {
        updateTargetColumnOptions(dataset.columns);
    }

    updatePreprocessingControls(true);
    const algorithmSelect = document.getElementById("algorithm-select");
    const taskSelect = document.getElementById("task-type-select");
    if (algorithmSelect) algorithmSelect.disabled = false;
    if (taskSelect) taskSelect.disabled = false;
    updateTrainAvailability();
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
            ? "Replacing…"
            : "Uploading…";
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
            ? "Validating and replacing the active dataset…"
            : "Validating and uploading your CSV…";
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
    status.textContent = "Loading dataset sample…";

    try {
        const preview = await apiRequest(`/dataset/${encodeURIComponent(datasetId)}/preview`);
        document.getElementById("dataset-meta").textContent =
            `${preview.row_count} rows · ${preview.column_count} columns`;

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
                cell.textContent = value === null || value === undefined ? "—" : String(value);
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
                        ? "—"
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

        await restoreActiveDataset();
    await loadComparisonRuns();

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

// ─── Administrative Control Center (US-14a / FR-7.1, 7.2, 7.3) ───────────────

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
