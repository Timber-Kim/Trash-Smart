const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const predictBtn = document.getElementById('predictBtn');
const resetBtn = document.getElementById('resetBtn');
const resultSection = document.getElementById('resultSection');
const resultCards = document.getElementById('resultCards');
const loading = document.getElementById('loading');

let selectedFile = null;

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('active'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('active'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('active');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) setFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

resetBtn.addEventListener('click', reset);

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  showLoading(true);
  resultSection.style.display = 'none';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      alert(data.detail || '예측 중 오류가 발생했습니다.');
      return;
    }
    renderResults(data.predictions);
  } catch (err) {
    alert('서버와 연결할 수 없습니다.');
  } finally {
    showLoading(false);
  }
});

function setFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    dropZone.style.display = 'none';
    previewContainer.style.display = 'block';
    predictBtn.disabled = false;
    resultSection.style.display = 'none';
  };
  reader.readAsDataURL(file);
}

function reset() {
  selectedFile = null;
  fileInput.value = '';
  dropZone.style.display = 'block';
  previewContainer.style.display = 'none';
  predictBtn.disabled = true;
  resultSection.style.display = 'none';
  resultCards.innerHTML = '';
}

function showLoading(show) {
  loading.style.display = show ? 'block' : 'none';
  predictBtn.disabled = show;
}

function renderResults(predictions) {
  resultCards.innerHTML = '';
  predictions.forEach((pred, i) => {
    const pct = Math.round(pred.confidence * 100);
    const card = document.createElement('div');
    card.className = `result-card rank-${i + 1}`;
    card.style.borderLeftColor = pred.color;
    card.innerHTML = `
      <div class="rank-badge">${i + 1}</div>
      <div class="result-info">
        <div class="result-name">${pred.name_ko}</div>
        <div class="result-name-en">${pred.name_en}</div>
        <div class="result-guide">💡 ${pred.disposal_guide}</div>
      </div>
      <div class="confidence-bar">
        <div class="confidence-pct">${pct}%</div>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${pred.color}"></div></div>
      </div>
    `;
    resultCards.appendChild(card);
  });
  resultSection.style.display = 'block';
  resultSection.scrollIntoView({ behavior: 'smooth' });
}
