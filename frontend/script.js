

const API_BASE = 'http://localhost:8000';
const gradeColors = { A: "#2e7d32", B: "#8bc34a", C: "#fbc02d", D: "#ef6c00", E: "#c62828" };

async function loadCart() {
  try {
    const res = await fetch(`${API_BASE}/cart`);
    if (!res.ok) throw new Error('Failed to load cart');
    const data = await res.json();
    renderCart(data.items || []);
  } catch (err) {
    console.error("Failed to load cart:", err);
  }
  
  try {
    const res2 = await fetch(`${API_BASE}/cart/analysis`);
    if (!res2.ok) throw new Error('Failed to load analysis');
    const analysis = await res2.json();
    renderSummary(analysis);
  } catch (err) {
    console.error("Failed to load analysis:", err);
  }
}

function renderCart(items) {
  const list = document.getElementById("cart-list");
  if (!list) return;
  
  const empty = document.getElementById("cart-empty");
  list.querySelectorAll(".cart-item").forEach(el => el.remove());

  if (!items || items.length === 0) {
    if (empty) empty.classList.remove("hidden");
    return;
  }
  if (empty) empty.classList.add("hidden");

  items.forEach(item => {
    const product = item.product || {};
    const grading = item.grading || {};
    const grade = grading.grade || "?";
    const color = gradeColors[grade] || "#999";

    const card = document.createElement("div");
    card.className = "cart-item glass-card";
    card.innerHTML = `
      <div class="cart-item-info">
        <p class="cart-item-name">${product.name || "Unknown product"}</p>
        <p class="cart-item-brand">${product.brand || ""}</p>
        <p class="cart-item-barcode">Barcode: ${product.barcode || ""}</p>
      </div>
      <div class="cart-item-grade" style="background:${color}">${grade}</div>
      <button class="cart-item-remove" data-barcode="${product.barcode}">✕</button>
    `;
    list.appendChild(card);
  });

  list.querySelectorAll(".cart-item-remove").forEach(btn => {
    btn.addEventListener("click", () => removeItem(btn.dataset.barcode));
  });
}

function renderSummary(analysis) {
  const summary = document.getElementById("cart-summary");
  if (!summary) return;
  
  if (!analysis || analysis.total_items === 0) {
    summary.classList.add("hidden");
    return;
  }
  summary.classList.remove("hidden");

  const gradeBits = Object.entries(analysis.grades || {})
    .filter(([, count]) => count > 0)
    .map(([grade, count]) => `<span class="grade-pill" style="background:${gradeColors[grade]}">${grade}: ${count}</span>`)
    .join("");

  summary.innerHTML = `
    <p class="summary-title">${analysis.total_items} item${analysis.total_items === 1 ? "" : "s"} in cart</p>
    <div class="grade-pills">${gradeBits}</div>
    ${analysis.low_rated_items && analysis.low_rated_items.length
      ? `<p class="summary-warning">⚠ ${analysis.low_rated_items.length} low-rated item(s)</p>` : ""}
  `;
}

async function removeItem(barcode) {
  try {
    await fetch(`${API_BASE}/cart/${barcode}`, {
      method: 'DELETE'
    });
    loadCart();
  } catch (err) {
    console.error("Failed to remove item:", err);
  }
}

// Load cart when page loads (if cart page)
if (document.getElementById("cart-list")) {
  loadCart();
}