/**
 * Live inference charts from /api/state inference.ticks
 */
window.InferenceCharts = (function () {
  const charts = {
    pnl: null,
    grid: null,
    load: null,
    cap: null,
  };

  let lastTickCount = 0;
  let lastJson = "";

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function formatMoney(cents) {
    if (!Number.isFinite(cents)) return "—";
    const dollars = cents / 100;
    const sign = dollars >= 0 ? "+" : "";
    return `${sign}$${Math.abs(dollars).toFixed(2)}`;
  }

  function formatMoneyPlain(cents) {
    if (!Number.isFinite(cents)) return "—";
    return `$${(cents / 100).toFixed(2)}`;
  }

  function baseOptions(yTitle, y2Title) {
    const opts = {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 450, easing: "easeOutQuart" },
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1a1814",
          titleFont: { family: "IBM Plex Sans", size: 12 },
          bodyFont: { family: "IBM Plex Mono", size: 11 },
          padding: 10,
          cornerRadius: 6,
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(184, 176, 162, 0.35)" },
          ticks: {
            maxTicksLimit: 10,
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
        y: {
          grid: { color: "rgba(184, 176, 162, 0.35)" },
          ticks: {
            font: { family: "IBM Plex Mono", size: 10 },
            color: "#8a847a",
          },
          title: {
            display: true,
            text: yTitle,
            font: { size: 11, weight: "500" },
            color: "#5c574f",
          },
        },
      },
    };

    if (y2Title) {
      opts.scales.y2 = {
        type: "linear",
        position: "right",
        grid: { drawOnChartArea: false },
        ticks: {
          font: { family: "IBM Plex Mono", size: 10 },
          color: "#8a847a",
        },
        title: {
          display: true,
          text: y2Title,
          font: { size: 11, weight: "500" },
          color: css("--cap-v"),
        },
      };
    }

    return opts;
  }

  function makeGradient(chart, positive) {
    const { chartArea, ctx } = chart;
    if (!chartArea) return positive ? css("--profit-soft") : css("--loss-soft");
    const g = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
    if (positive) {
      g.addColorStop(0, "rgba(45, 106, 79, 0.38)");
      g.addColorStop(1, "rgba(45, 106, 79, 0.02)");
    } else {
      g.addColorStop(0, "rgba(163, 59, 43, 0.32)");
      g.addColorStop(1, "rgba(163, 59, 43, 0.02)");
    }
    return g;
  }

  function cumulativePnl(ticks) {
    let sum = 0;
    return ticks.map((t) => {
      sum += Number(t.tick_profit_cents) || 0;
      return sum;
    });
  }

  function ensurePnlChart(labels, data) {
    const canvas = document.getElementById("pnl-chart");
    if (!canvas) return;

    const last = data.length ? data[data.length - 1] : 0;
    const positive = last >= 0;
    const color = positive ? css("--profit") : css("--loss");

    if (charts.pnl) {
      charts.pnl.data.labels = labels;
      charts.pnl.data.datasets[0].data = data;
      charts.pnl.data.datasets[0].borderColor = color;
      charts.pnl.update("none");
      if (charts.pnl.chartArea) {
        charts.pnl.data.datasets[0].backgroundColor = makeGradient(charts.pnl, positive);
      }
      return;
    }

    charts.pnl = new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Cumulative P&L (¢)",
            data,
            borderColor: color,
            backgroundColor: css("--profit-soft"),
            fill: true,
            tension: 0.42,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 2.5,
          },
        ],
      },
      options: {
        ...baseOptions("¢ cumulative"),
        plugins: {
          ...baseOptions("¢ cumulative").plugins,
          tooltip: {
            ...baseOptions("¢ cumulative").plugins.tooltip,
            callbacks: {
              label(ctx) {
                return ` ${formatMoney(ctx.parsed.y)}`;
              },
            },
          },
        },
      },
    });
  }

  function ensureLineChart(key, canvasId, label, data, color, yTitle, fillSoft) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    if (charts[key]) {
      charts[key].data.labels = data.labels;
      charts[key].data.datasets[0].data = data.values;
      charts[key].update("none");
      return;
    }

    charts[key] = new Chart(canvas, {
      type: "line",
      data: {
        labels: data.labels,
        datasets: [
          {
            label,
            data: data.values,
            borderColor: color,
            backgroundColor: fillSoft,
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 3,
            borderWidth: 2,
          },
        ],
      },
      options: baseOptions(yTitle),
    });
  }

  function ensureCapChart(labels, pValues, vValues) {
    const canvas = document.getElementById("cap-chart");
    if (!canvas) return;

    if (charts.cap) {
      charts.cap.data.labels = labels;
      charts.cap.data.datasets[0].data = pValues;
      charts.cap.data.datasets[1].data = vValues;
      charts.cap.update("none");
      return;
    }

    charts.cap = new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Cap power (W)",
            data: pValues,
            borderColor: css("--cap-p"),
            backgroundColor: css("--grid-soft"),
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            borderWidth: 2,
            yAxisID: "y",
          },
          {
            label: "Cap voltage (V)",
            data: vValues,
            borderColor: css("--cap-v"),
            backgroundColor: "transparent",
            fill: false,
            tension: 0.4,
            pointRadius: 0,
            borderWidth: 1.75,
            borderDash: [4, 3],
            yAxisID: "y2",
          },
        ],
      },
      options: baseOptions("W", "V"),
    });
  }

  function updateHero(summary, ticks) {
    const pnlEl = document.getElementById("inference-pnl-total");
    const pnlSub = document.getElementById("inference-pnl-sub");
    const tickEl = document.getElementById("inference-pnl-tick");
    const countEl = document.getElementById("inference-tick-count");
    const badge = document.getElementById("inference-live-badge");
    const section = document.getElementById("inference-section");
    const pnlBox = document.querySelector(".inference-hero__stat--pnl");

    const cumulative = Number(summary?.cumulative_profit_cents) || 0;
    const lastTick = ticks.length ? ticks[ticks.length - 1] : null;
    const lastProfit = lastTick ? Number(lastTick.tick_profit_cents) : null;
    const active = Boolean(summary?.active);

    if (pnlEl) {
      pnlEl.textContent = formatMoney(cumulative);
      pnlEl.classList.toggle("is-loss", cumulative < 0);
    }
    if (pnlBox) {
      pnlBox.classList.toggle("inference-hero__stat--loss", cumulative < 0);
    }
    if (pnlSub) {
      pnlSub.textContent = active
        ? `Day ${summary.last_day ?? "—"} · tick ${summary.last_tick ?? "—"}`
        : "Start inference.py to begin logging";
    }
    if (tickEl) {
      tickEl.textContent = lastProfit === null ? "—" : formatMoney(lastProfit);
      tickEl.classList.toggle("is-loss", lastProfit !== null && lastProfit < 0);
    }
    if (countEl) {
      countEl.textContent = String(summary?.tick_count ?? 0);
    }
    if (badge) {
      badge.textContent = active ? "Live" : "Standby";
      badge.classList.toggle("is-standby", !active);
    }
    if (section) {
      section.classList.toggle("is-inactive", !active);
    }
  }

  function render(inference) {
    const ticks = inference?.ticks ?? [];
    const summary = inference?.summary ?? {};
    const json = JSON.stringify(ticks);

    updateHero(summary, ticks);

    if (json === lastJson) return;
    lastJson = json;

    if (!ticks.length) return;

    const labels = ticks.map((t) => t.tick);
    const pnl = cumulativePnl(ticks);

    ensurePnlChart(labels, pnl);

    ensureLineChart(
      "grid",
      "grid-chart",
      "Grid import",
      {
        labels,
        values: ticks.map((t) => t.grid_import_w ?? 0),
      },
      css("--grid"),
      "W",
      css("--grid-soft")
    );

    ensureLineChart(
      "load",
      "load-chart",
      "Total demand",
      {
        labels,
        values: ticks.map((t) => t.total_demand_w ?? 0),
      },
      css("--load"),
      "W",
      css("--load-soft")
    );

    const pValues = ticks.map((t) => {
      if (t.pcout_w != null && Number.isFinite(Number(t.pcout_w))) return Number(t.pcout_w);
      return Number(t.sc_bus_w) || 0;
    });
    const vValues = ticks.map((t) =>
      t.vcap_v != null && Number.isFinite(Number(t.vcap_v)) ? Number(t.vcap_v) : null
    );

    ensureCapChart(labels, pValues, vValues);
    lastTickCount = ticks.length;
  }

  return { render };
})();
