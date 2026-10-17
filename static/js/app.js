// PulseGuard AI - Client JS & Modal Engine

const PRESETS = {
  acs: {
    name: "Ramesh Sharma",
    age: 62,
    sex: "Male",
    bp_systolic: 145,
    bp_diastolic: 92,
    pulse: 104,
    temp_c: 37.1,
    spo2: 95,
    chief_complaint: "Severe crushing chest pain radiating to left arm and jaw with profuse sweating for 1 hour",
    known_conditions: "Hypertension, CAD with stent in 2021",
    medication_list: "Aspirin, Atorvastatin, Amlodipine"
  },
  stroke: {
    name: "Lakshmi Devi",
    age: 68,
    sex: "Female",
    bp_systolic: 175,
    bp_diastolic: 102,
    pulse: 88,
    temp_c: 36.8,
    spo2: 97,
    chief_complaint: "Sudden facial droop, slurred speech and right arm weakness since 45 minutes",
    known_conditions: "Diabetes Type 2, Hypertension",
    medication_list: "Metformin, Telmisartan"
  },
  resp_distress: {
    name: "Mohammad Ali",
    age: 48,
    sex: "Male",
    bp_systolic: 130,
    bp_diastolic: 85,
    pulse: 122,
    temp_c: 38.8,
    spo2: 87, // Critical hypoxia
    chief_complaint: "Severe breathlessness, cannot speak full sentences, wheezing and cough for 6 hours",
    known_conditions: "Asthma, Chronic Bronchitis",
    medication_list: "Salbutamol inhaler"
  },
  abstain_vitals: {
    name: "Suresh Reddy",
    age: 52,
    sex: "Male",
    bp_systolic: 0, // Corrupt zero BP
    bp_diastolic: 0,
    pulse: 0, // Corrupt zero pulse
    temp_c: 96.0, // Implausible / Fahrenheit
    spo2: 94,
    chief_complaint: "Feeling dizzy and uneasy since morning",
    known_conditions: "None reported",
    medication_list: "None"
  },
  pediatric_fever: {
    name: "Baby Ananya",
    age: 1,
    sex: "Female",
    bp_systolic: 80,
    bp_diastolic: 50,
    pulse: 165,
    temp_c: 102.2, // 39.0 C Fahrenheit
    spo2: 96,
    chief_complaint: "High fever 2 days, not drinking milk, very lethargic and crying weakly",
    known_conditions: "None",
    medication_list: "Paracetamol drops"
  },
  mild_cold: {
    name: "Vikram Kumar",
    age: 26,
    sex: "Male",
    bp_systolic: 118,
    bp_diastolic: 78,
    pulse: 72,
    temp_c: 36.6,
    spo2: 99,
    chief_complaint: "Runny nose, mild throat irritation and sneezing for 3 days, denies chest pain or fever",
    known_conditions: "None",
    medication_list: "None"
  }
};

function syncBpFields(val) {
  const sysEl = document.getElementById("bp_systolic");
  const diaEl = document.getElementById("bp_diastolic");
  if (!sysEl || !diaEl) return;
  if (!val || !val.trim()) {
    sysEl.value = "";
    diaEl.value = "";
    return;
  }
  const parts = val.trim().split(/[/,\-\s]+/);
  if (parts.length >= 2) {
    sysEl.value = parts[0].trim();
    diaEl.value = parts[1].trim();
  } else if (parts.length === 1 && !isNaN(parts[0])) {
    sysEl.value = parts[0].trim();
    diaEl.value = "";
  }
}

function loadPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.value = val !== undefined && val !== null ? val : "";
  };
  setVal("name", p.name);
  setVal("age", p.age);
  setVal("sex", p.sex);
  
  // Set unified BP input and hidden components
  const bpFormatted = (p.bp_systolic !== undefined && p.bp_diastolic !== undefined) 
    ? `${p.bp_systolic}/${p.bp_diastolic}` 
    : "";
  setVal("bp", bpFormatted);
  setVal("bp_systolic", p.bp_systolic);
  setVal("bp_diastolic", p.bp_diastolic);

  setVal("pulse", p.pulse);
  setVal("temp_c", p.temp_c);
  setVal("spo2", p.spo2);
  setVal("chief_complaint", p.chief_complaint);
  setVal("known_conditions", p.known_conditions);
  setVal("medication_list", p.medication_list);
}

// Table Filter
function filterQueue() {
  const deptFilter = document.getElementById("deptFilter") ? document.getElementById("deptFilter").value.toLowerCase() : "all";
  const searchInput = document.getElementById("searchInput") ? document.getElementById("searchInput").value.toLowerCase() : "";
  const rows = document.querySelectorAll("#queueTableBody tr");

  rows.forEach(row => {
    const dept = row.getAttribute("data-dept") ? row.getAttribute("data-dept").toLowerCase() : "";
    const text = row.innerText.toLowerCase();
    
    const matchesDept = (deptFilter === "all" || dept === deptFilter);
    const matchesSearch = (!searchInput || text.includes(searchInput));

    if (matchesDept && matchesSearch) {
      row.style.display = "";
    } else {
      row.style.display = "none";
    }
  });
}

// =========================================================
// Interactive Review & Override Modal Logic
// =========================================================
let currentOverridePatientId = null;

function openOverrideModal(p) {
  currentOverridePatientId = p.patientId;
  const modal = document.getElementById("overrideModal");
  if (!modal) {
    // If modal not found on page, fallback to detail page navigation
    window.location.href = "/patient/" + p.patientId;
    return;
  }

  // Populate header and summary
  document.getElementById("modalPatientName").innerText = p.name;
  document.getElementById("modalPatientMeta").innerText = `${p.patientId} • ${p.age}y / ${p.sex}`;
  document.getElementById("modalVitalsBP").innerText = p.bp || "--";
  document.getElementById("modalVitalsHR").innerText = (p.pulse || "--") + " bpm";
  document.getElementById("modalVitalsSpO2").innerText = (p.spo2 || "--") + "%";
  document.getElementById("modalVitalsTemp").innerText = (p.temp || "--") + "°C";
  document.getElementById("modalComplaint").innerText = `"${p.complaint}"`;
  
  // Set AI details
  document.getElementById("modalAiRecommendation").innerText = `Level ${p.finalLevel || p.aiLevel} (${p.finalDept || p.aiDept})`;

  // Red flags
  const rfContainer = document.getElementById("modalRedFlagsContainer");
  rfContainer.innerHTML = "";
  if (p.redFlags && p.redFlags.length > 0) {
    p.redFlags.forEach(rf => {
      const span = document.createElement("span");
      span.style = "display: inline-block; background-color: #fee2e2; color: #991b1b; font-size: 0.75rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 4px; margin-right: 0.35rem; margin-top: 0.25rem;";
      span.innerText = "⚠ " + rf;
      rfContainer.appendChild(span);
    });
  } else {
    rfContainer.innerHTML = "<span style='color: #64748b; font-size: 0.85rem;'>No critical red flags triggered.</span>";
  }

  // Form inputs
  document.getElementById("modal_patient_id").value = p.patientId;
  document.getElementById("modal_assessment_id").value = p.assessmentId || "";
  document.getElementById("modal_new_level").value = p.finalLevel || "3";
  document.getElementById("modal_new_dept").value = p.finalDept || "General Medicine";
  document.getElementById("modal_override_reason").value = "";

  // Detail link
  const detailLink = document.getElementById("modalDetailLink");
  if (detailLink) {
    detailLink.href = "/patient/" + p.patientId;
  }

  // Show modal
  modal.style.display = "flex";
  document.body.style.overflow = "hidden";
}

function closeOverrideModal() {
  const modal = document.getElementById("overrideModal");
  if (modal) {
    modal.style.display = "none";
  }
  document.body.style.overflow = "auto";
}

// Close when clicking outside modal card
window.addEventListener("click", function(e) {
  const modal = document.getElementById("overrideModal");
  if (e.target === modal) {
    closeOverrideModal();
  }
});

// Modal Override Submission via AJAX
function submitModalOverride(e) {
  e.preventDefault();
  const patientId = document.getElementById("modal_patient_id").value;
  const assessmentId = document.getElementById("modal_assessment_id").value;
  const nurseName = document.getElementById("modal_nurse_name").value.trim() || "Staff Nurse";
  const newLevel = parseInt(document.getElementById("modal_new_level").value);
  const newDept = document.getElementById("modal_new_dept").value;
  const reason = document.getElementById("modal_override_reason").value.trim();

  if (!reason || reason.length < 5) {
    alert("Clinical Policy Warning: A detailed clinical justification (minimum 5 characters) is required to override the AI recommendation.");
    document.getElementById("modal_override_reason").focus();
    return false;
  }

  const submitBtn = document.getElementById("modalSubmitBtn");
  const origBtnText = submitBtn.innerText;
  submitBtn.disabled = true;
  submitBtn.innerText = "Saving Override...";

  fetch("/override", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Requested-With": "XMLHttpRequest"
    },
    body: JSON.stringify({
      patient_id: patientId,
      assessment_id: assessmentId,
      nurse_name: nurseName,
      new_level: newLevel,
      new_dept: newDept,
      override_reason: reason
    })
  })
  .then(res => res.json())
  .then(data => {
    submitBtn.disabled = false;
    submitBtn.innerText = origBtnText;

    if (data.status === "success") {
      closeOverrideModal();
      // Reload page to reflect updated queue ordering and badges
      window.location.reload();
    } else {
      alert("Error: " + (data.message || "Failed to save override."));
    }
  })
  .catch(err => {
    submitBtn.disabled = false;
    submitBtn.innerText = origBtnText;
    // Fallback: submit standard form if AJAX fetch encounters an issue
    document.getElementById("modalOverrideForm").submit();
  });

  return false;
}

// Form validation for dedicated Detail Page Nurse Override
function validateOverrideForm(e) {
  const reasonEl = document.getElementById("override_reason");
  const reason = reasonEl ? reasonEl.value.trim() : "";
  if (!reason || reason.length < 5) {
    alert("Clinical Policy Warning: A detailed clinical justification (minimum 5 characters) is required to override the AI triage recommendation.");
    if (e && e.preventDefault) e.preventDefault();
    if (reasonEl) reasonEl.focus();
    return false;
  }
  return true;
}
