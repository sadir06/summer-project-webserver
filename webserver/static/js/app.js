const POLL_MS = 1000;
const CYCLE_TICKS = 59;
const CIRCUMFERENCE = 2 * Math.PI * 48;

const els = {
  connectionStatus: document.getElementById("connection-status"),
  headerTick: document.getElementById("header-tick"),
  sunValue: document.getElementById("sun-value"),
  sunBar: document.getElementById("sun-bar"),
  vcapValue: document.getElementById("vcap-bottom-value"),
  pcapValue: document.getElementById("pcout-bottom-value"),
  demandValue: document.getElementById("demand-value"),
  buyPriceValue: document.getElementById("buy-price-value"),
  sellPriceValue: document.getElementById("sell-price-value"),
  deferablesCount: document.getElementById("deferables-count"),
  deferablesList: document.getElementById("deferables-list"),
  cycleTick: document.getElementById("cycle-tick"),
  cycleProgress: document.getElementById("cycle-progress"),
  dayId: document.getElementById("day-id"),
  sunTick: document.getElementById("sun-tick"),
  priceTick: document.getElementById("price-tick"),
  demandTick: document.getElementById("demand-tick"),
  lastUpdated: document.getElementById("last-updated"),
  historyChart: document.getElementById("history-chart"),
};

let historyChart = null;
let lastYesterdayJson = "";

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatInt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString();
}

function flash(el) {
  if (!el) return;
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
}

function setConnectionState(ok, message) {
  els.connectionStatus.classList.toggle("status-pill--live", ok);
  els.connectionStatus.classList.toggle("status-pill--error", !ok);
  els.connectionStatus.lastChild.textContent = message;
}

function setText(el, next, flashOnChange = true) {
  if (!el) return;
  const prev = el.textContent;
  el.textContent = next;
  if (flashOnChange && prev !== "—" && prev !== next) flash(el);
}

function updateCycleDial(tick) {
  const safeTick = Number.isFinite(tick) ? Math.max(0, Math.min(CYCLE_TICKS, tick)) : null;
  setText(els.cycleTick, safeTick === null ? "—" : String(safeTick), false);

  if (safeTick === null) {
    els.cycleProgress.style.strokeDashoffset = CIRCUMFERENCE;
    return;
  }

  const progress = safeTick / CYCLE_TICKS;
  els.cycleProgress.style.strokeDashoffset = CIRCUMFERENCE * (1 - progress);
}

function renderDeferables(deferables, currentTick) {
  if (!Array.isArray(deferables) || deferables.length === 0) {
    els.deferablesCount.textContent = "0 tasks";
    els.deferablesList.innerHTML = '<p class="empty-state">No deferrable loads this cycle</p>';
    return;
  }

  els.deferablesCount.textContent = `${deferables.length} task${deferables.length === 1 ? "" : "s"}`;

  els.deferablesList.innerHTML = deferables
    .map((item, index) => {
      const start = Number(item.start);
      const end = Number(item.end);
      const energy = Number(item.energy);

      const left = Number.isFinite(start) ? (start / CYCLE_TICKS) * 100 : 0;
      const width =
        Number.isFinite(start) && Number.isFinite(end)
          ? Math.max(((end - start) / CYCLE_TICKS) * 100, 1.5)
          : 1.5;

      const nowLeft = Number.isFinite(currentTick) ? (currentTick / CYCLE_TICKS) * 100 : null;

      return `
        <div class="deferable-row">
          <div class="deferable-row__energy">
            ${formatNumber(energy, 1)}
            <span>Wh</span>
          </div>
          <div class="deferable-row__track-wrap" aria-label="Task ${index + 1}, ticks ${start} to ${end}">
            <div class="deferable-row__track"></div>
            <div class="deferable-row__window" style="left:${left}%; width:${width}%"></div>
            ${nowLeft !== null ? `<div class="deferable-row__now" style="left:${nowLeft}%"></div>` : ""}
          </div>
          <div class="deferable-row__range">${start} → ${end}</div>
        </div>
      `;
    })
    .join("");
}

function buildHistoryChart(yesterday) {
  const labels = yesterday.map((row) => row.tick);
  const demand = yesterday.map((row) => row.demand);
  const buy = yesterday.map((row) => row.buy_price);
  const sell = yesterday.map((row) => row.sell_price);

  const styles = getComputedStyle(document.documentElement);

  if (historyChart) {
    historyChart.data.labels = labels;
    historyChart.data.datasets[0].data = demand;
    historyChart.data.datasets[1].data = buy;
    historyChart.data.datasets[2].data = sell;
    historyChart.update("none");
    return;
  }

  historyChart = new Chart(els.historyChart, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Demand (W)",
          data: demand,
          borderColor: styles.getPropertyValue("--demand").trim(),
          backgroundColor: "rgba(47, 95, 143, 0.08)",
          yAxisID: "yDemand",
          tension: 0.25,
          pointRadius: 0,
          borderWidth: 2,
          fill: true,
        },
        {
          label: "Buy price",
          data: buy,
          borderColor: styles.getPropertyValue("--buy").trim(),
          yAxisID: "yPrice",
          tension: 0.25,
          pointRadius: 0,
          borderWidth: 1.5,
        },
        {
          label: "Sell price",
          data: sell,
          borderColor: styles.getPropertyValue("--sell").trim(),
          yAxisID: "yPrice",
          tension: 0.25,
          pointRadius: 0,
          borderWidth: 1.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1a1814",
          titleFont: { family: "IBM Plex Sans" },
          bodyFont: { family: "IBM Plex Mono", size: 12 },
          padding: 10,
          cornerRadius: 6,
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(184, 176, 162, 0.35)" },
          ticks: {
            maxTicksLimit: 12,
            font: { family: "IBM Plex Mono", size: 10 },
            color: "#8a847a",
          },
          title: {
            display: true,
            text: "Tick",
            font: { size: 11, weight: "500" },
            color: "#5c574f",
          },
        },
        yDemand: {
          type: "linear",
          position: "left",
          grid: { color: "rgba(184, 176, 162, 0.35)" },
          ticks: {
            font: { family: "IBM Plex Mono", size: 10 },
            color: "#8a847a",
          },
          title: {
            display: true,
            text: "W",
            font: { size: 11, weight: "500" },
            color: "#2f5f8f",
          },
        },
        yPrice: {
          type: "linear",
          position: "right",
          grid: { drawOnChartArea: false },
          ticks: {
            font: { family: "IBM Plex Mono", size: 10 },
            color: "#8a847a",
          },
          title: {
            display: true,
            text: "Price",
            font: { size: 11, weight: "500" },
            color: "#5c574f",
          },
        },
      },
    },
  });
}

function renderState(state) {
  const pvout = Number(state.pvout);
  const vcap = Number(state.vcap);
  const pcap = Number(state.pcout);
  const price = state.price ?? {};
  const demand = state.demand ?? {};
  const deferables = state.deferables ?? [];
  const yesterday = state.yesterday ?? [];

  // Display Pico pvout where sun used to be displayed
  setText(els.sunValue, Number.isFinite(pvout) ? `${formatNumber(pvout, 3)} W` : "—");

  // Bar assumes max PV panel power is 8 W
  els.sunBar.style.width = Number.isFinite(pvout)
    ? `${Math.min(Math.max(pvout / 8, 0), 1) * 100}%`
    : "0%";

  setText(els.vcapValue, Number.isFinite(vcap) ? `${formatNumber(vcap, 3)} V` : "—");
  setText(els.pcapValue, Number.isFinite(pcap) ? `${formatNumber(pcap, 3)} W` : "—");

  setText(
    els.demandValue,
    Number.isFinite(Number(demand.demand)) ? formatNumber(demand.demand, 2) : "—"
  );

  setText(
    els.buyPriceValue,
    Number.isFinite(Number(price.buy_price)) ? formatInt(price.buy_price) : "—"
  );

  setText(
    els.sellPriceValue,
    Number.isFinite(Number(price.sell_price)) ? formatInt(price.sell_price) : "—"
  );

  // Use demand or price tick. Sun has been removed.
  const primaryTick = [demand.tick, price.tick].find((t) => Number.isFinite(Number(t)));
  const tickNum = Number(primaryTick);

  setText(els.headerTick, Number.isFinite(tickNum) ? String(tickNum) : "—", false);
  updateCycleDial(Number.isFinite(tickNum) ? tickNum : null);

  setText(els.dayId, price.day ?? demand.day ?? "—", false);

  // Reuse the old sun tick field to show PV output source/status
  setText(els.sunTick, Number.isFinite(pvout) ? "PV" : "—", false);

  setText(els.priceTick, price.tick ?? "—", false);
  setText(els.demandTick, demand.tick ?? "—", false);

  renderDeferables(deferables, Number.isFinite(tickNum) ? tickNum : null);

  const yesterdayJson = JSON.stringify(yesterday);
  if (Array.isArray(yesterday) && yesterday.length > 0 && yesterdayJson !== lastYesterdayJson) {
    lastYesterdayJson = yesterdayJson;
    buildHistoryChart(yesterday);
  }

  if (state.last_updated) {
    const date = new Date(state.last_updated);
    els.lastUpdated.textContent = Number.isNaN(date.getTime())
      ? state.last_updated
      : date.toLocaleTimeString();
    els.lastUpdated.dateTime = state.last_updated;
  }
}

async function poll() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const state = await response.json();

    console.log("STATE:", state);
    console.log("PVOUT:", state.pvout);
    console.log("VCAP:", state.vcap);
    console.log("PCOUT:", state.pcout);

    renderState(state);
    if (window.InferenceCharts && state.inference) {
      window.InferenceCharts.render(state.inference);
    }
    setConnectionState(true, "Live");
  } catch (err) {
    setConnectionState(false, "Offline");
    console.error("Failed to fetch /api/state:", err);
  }
}

poll();
setInterval(poll, POLL_MS);