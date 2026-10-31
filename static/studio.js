(() => {
  "use strict";
  const $ = (selector, root=document) => root.querySelector(selector);
  const $$ = (selector, root=document) => [...root.querySelectorAll(selector)];
  const page = document.body.dataset.page || "dashboard";
  const useGpuAcceleration = $("#useGpuAcceleration");
  const computeBackendWrap = $("#computeBackendWrap");
  const computeBackend = $("#computeBackend");
  const gpuStatus = $("#gpuStatus");
  const gpuModePill = $("#gpuModePill");
  let computeAvailability = null;
  let profiles = [];
  const assetCache = new Map();
  const generatedPhotoAssetCache = new Map();
  const selectedGeneratedAssetIds = new Set();
  const generatedPhotoLimit = 18;
  let generatedPhotoOffset = 0;
  let generatedPhotoTotal = 0;
  const assetPageSize = 30;
  let assetOffset = 0;
  let assetTotal = 0;

  async function readJson(response) {
    let data = {};
    try { data = await response.json(); } catch (_) {}
    if (!response.ok) throw new Error(data.error || data.message || ("Request failed: HTTP " + response.status));
    return data;
  }
  function addComputeData(formData) {
    if (useGpuAcceleration) {
      formData.set("use_gpu", useGpuAcceleration.checked ? "true" : "false");
      formData.set("compute_backend", useGpuAcceleration.checked ? (computeBackend?.value || "auto") : "cpu");
    }
    formData.delete("use_runpod");
    return formData;
  }
  function backendLabel(key, options) {
    if (key === "auto") return "Auto · use available GPU";
    if (key === "local_gpu") {
      const gpu = options.local_gpu || {};
      return "System GPU" + (gpu.name ? " · " + gpu.name : "") + (gpu.vram_gb ? " · " + gpu.vram_gb + " GB VRAM" : "");
    }
    if (key === "cloud_gpu" || key === "runpod") {
      const gpu = options.cloud_gpu || {};
      return "GHive / Colab GPU" + (gpu.name ? " · " + gpu.name : "") + (gpu.vram_gb ? " · " + gpu.vram_gb + " GB VRAM" : "");
    }
    return "Local CPU (fallback)";
  }
  function applyComputeStatus() {
    if (!useGpuAcceleration || !gpuStatus || !gpuModePill) return;
    const requested = useGpuAcceleration.checked ? (computeBackend?.value || "auto") : "cpu";
    if (!useGpuAcceleration.checked) {
      gpuModePill.textContent = "LOCAL · CPU";
      gpuStatus.textContent = "GPU acceleration is off. Requests use the local CPU; no cloud request is made.";
      return;
    }
    const selected = requested === "auto" ? (computeAvailability?.auto_backend || "auto") : requested;
    const labels = {cpu:"CPU FALLBACK", local_gpu:"SYSTEM GPU", cloud_gpu:"REMOTE GPU", runpod:"REMOTE GPU", auto:"GPU UNAVAILABLE"};
    gpuModePill.textContent = requested === "auto" ? "AUTO · " + (labels[selected] || "CHECKING GPU") : (labels[selected] || "GPU ENABLED");
    if (selected === "local_gpu") {
      gpuStatus.textContent = "GPU acceleration is ON. Image generation will run on the detected System GPU: " + (computeAvailability?.local_gpu?.name || "CUDA GPU") + ".";
    } else if (selected === "cloud_gpu" || selected === "runpod") {
      gpuStatus.textContent = "GPU acceleration is ON. Generation will run on the connected remote GPU: " + (computeAvailability?.cloud_gpu?.name || "GHive / Colab GPU") + ". Requests are sent securely from the app to your configured worker.";
    } else if (requested === "auto") {
      gpuStatus.textContent = "GPU acceleration is ON, but no usable GPU is available. Auto mode is falling back to local CPU; image generation may be very slow. " + (computeAvailability?.cloud_error || "Check the cloud worker URL, shared token, and running runtime.");
    } else {
      gpuStatus.textContent = "CPU fallback selected explicitly. Generation will run on the local CPU, not a GPU.";
    }
    if (computeAvailability?.cloud_error && selected !== "cloud_gpu" && selected !== "runpod") gpuStatus.textContent += " GHive / Colab GPU unavailable: " + computeAvailability.cloud_error;
  }
  function renderComputeOptions(options) {
    computeAvailability = options;
    if (!computeBackend) return;
    const previous = computeBackend.value;
    computeBackend.replaceChildren();
    const autoOption = document.createElement("option");
    autoOption.value = "auto";
    autoOption.textContent = backendLabel("auto", options);
    computeBackend.appendChild(autoOption);
    (options.available_backends || ["cpu"]).forEach(key => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = backendLabel(key, options);
      computeBackend.appendChild(option);
    });
    const choices = ["auto", ...(options.available_backends || ["cpu"])];
    computeBackend.value = choices.includes(previous) ? previous : "auto";
    applyComputeStatus();
  }
  async function probeCloudGpu() {
    if (!useGpuAcceleration || !computeBackendWrap) return;
    computeBackendWrap.hidden = !useGpuAcceleration.checked;
    if (!useGpuAcceleration.checked) {
      computeAvailability = null;
      renderComputeOptions({available_backends:["cpu"], auto_backend:"cpu", local_gpu:{}, cloud_gpu:{}, cloud_error:null});
      return;
    }
    if (gpuModePill) gpuModePill.textContent = "CHECKING GPUS…";
    if (gpuStatus) gpuStatus.textContent = "Checking local CUDA support and testing the configured remote GPU worker…";
    if (computeBackend) computeBackend.disabled = true;
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 25000);
    try {
      const response = await fetch("/api/gpu-status?probe=1", {
        cache: "no-store",
        signal: controller.signal,
        headers: { "Accept": "application/json" }
      });
      const data = await readJson(response);
      renderComputeOptions(data);
    } catch (error) {
      const message = error.name === "AbortError"
        ? "GPU check timed out after 25 seconds. Verify the Flask app, GPU_WORKER_URL, and worker token, then retry."
        : error.message || "Could not check GPU availability.";
      renderComputeOptions({
        available_backends: ["cpu"],
        auto_backend: "cpu",
        local_gpu: {available:false},
        cloud_gpu: {available:false},
        cloud_error: message
      });
    } finally {
      window.clearTimeout(timeoutId);
      if (computeBackend) computeBackend.disabled = false;
    }
  }
  if (useGpuAcceleration) {
    useGpuAcceleration.addEventListener("change", () => {
      computeBackendWrap.hidden = !useGpuAcceleration.checked;
      if (useGpuAcceleration.checked && computeBackend) computeBackend.value = "auto";
      else if (computeBackend) computeBackend.value = "cpu";
      applyComputeStatus();
      probeCloudGpu();
    });
    computeBackend?.addEventListener("change", applyComputeStatus);
    computeBackendWrap.hidden = !useGpuAcceleration.checked;
    applyComputeStatus();
    if (useGpuAcceleration.checked) probeCloudGpu();
  }

  function buildProfileOptions(select, firstLabel, selected) {
    if (!select) return;
    select.replaceChildren();
    const first = document.createElement("option");
    first.value = "";
    first.textContent = firstLabel;
    select.appendChild(first);
    profiles.forEach(profile => {
      const option = document.createElement("option");
      option.value = profile.id;
      option.textContent = profile.name + " · " + profile.image_count + " photo(s)"
        + (profile.local_model_ready ? " · local trained" : "")
        + (profile.runpod_model_ready ? " · Cloud trained" : "");
      select.appendChild(option);
    });
    if (profiles.some(profile => profile.id === selected)) select.value = selected;
  }
  function updateFaceConditionControls() {
    const profileSelected=Boolean($("#faceProfile")?.value);
    ["faceAngle","faceExpression"].forEach(id=>{
      const control=$("#"+id);
      if(!control)return;
      control.disabled=!profileSelected;
      if(!profileSelected)control.value="";
    });
  }
  $("#faceProfile")?.addEventListener("change",updateFaceConditionControls);

  function makeTag(text, state) {
    const node = document.createElement("span");
    node.className = "tag " + (state ? "ready" : "pending");
    node.textContent = text;
    return node;
  }
  function renderProfiles() {
    buildProfileOptions($("#trainProfile"), "Select a face profile", $("#trainProfile")?.value || "");
    buildProfileOptions($("#faceProfile"), "Base model (no trained face)", $("#faceProfile")?.value || "");
    buildProfileOptions($("#videoProfile"), "Base model (no trained face)", $("#videoProfile")?.value || "");
    buildProfileOptions($("#addPhotosProfile"), "Select a face profile", $("#addPhotosProfile")?.value || "");
    const profileList = $("#profileList");
    if (profileList) {
      profileList.replaceChildren();
      const count = $("#profileCount");
      if (count) count.textContent = profiles.length + (profiles.length === 1 ? " profile" : " profiles");
      if (!profiles.length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "No face profiles yet. Create your first profile.";
        profileList.appendChild(empty);
      }
      profiles.forEach(profile => {
        const card = document.createElement("article");
        card.className = "profile-card";
        const top = document.createElement("div"); top.className = "profile-card-top";
        const titleWrap = document.createElement("div");
        const title = document.createElement("div"); title.className = "profile-name"; title.textContent = profile.name;
        const meta = document.createElement("div"); meta.className = "profile-meta";
        meta.textContent = profile.image_count + " photo(s) · Token: " + profile.token;
        titleWrap.append(title, meta); top.appendChild(titleWrap);
        const countTag = makeTag(profile.image_count + " photos", profile.image_count >= 5); top.appendChild(countTag);
        const badges = document.createElement("div"); badges.className = "badge-row";
        badges.append(makeTag("Local: " + (profile.local_model_ready ? "trained" : (profile.local_training_state || "not trained")), profile.local_model_ready));
        badges.append(makeTag("Cloud: " + (profile.runpod_model_ready ? "trained" : (profile.runpod_training_state || "not trained")), profile.runpod_model_ready));
        const actions = document.createElement("div"); actions.className = "profile-actions";
        const useButton = document.createElement("button"); useButton.type="button"; useButton.className="secondary-button"; useButton.textContent="Use for training";
        useButton.addEventListener("click", () => {
          if ($("#trainProfile")) $("#trainProfile").value = profile.id;
          if ($("#addPhotosProfile")) $("#addPhotosProfile").value = profile.id;
          if ($("#faceProfile")) $("#faceProfile").value = profile.id;
          if ($("#videoProfile")) $("#videoProfile").value = profile.id;
          updateFaceConditionControls();
          $("#trainProfile")?.scrollIntoView({behavior:"smooth", block:"center"});
        });
        const deleteButton = document.createElement("button"); deleteButton.type="button"; deleteButton.className="danger-button"; deleteButton.textContent="Delete";
        deleteButton.addEventListener("click", async () => {
          if (!confirm("Delete face profile “" + profile.name + "”, its dataset, and local LoRA? Generated media is kept.")) return;
          deleteButton.disabled = true;
          try {
            await readJson(await fetch("/api/face-profiles/" + profile.id, {method:"DELETE"}));
            await refreshProfiles();
          } catch (error) { alert(error.message); }
          finally { deleteButton.disabled = false; }
        });
        actions.append(useButton, deleteButton);
        card.append(top, badges, actions); profileList.appendChild(card);
      });
    }
    loadGeneratedPhotoPicker();
  }
  async function refreshProfiles() {
    const response = await fetch("/api/face-profiles", {cache:"no-store", headers:{"Accept":"application/json"}});
    const data = await readJson(response);
    profiles = data.profiles || [];
    renderProfiles();
    const query = new URLSearchParams(window.location.search);
    const requestedProfileId = query.get("profile_id");
    if (requestedProfileId && profiles.some(profile => profile.id === requestedProfileId)) {
      ["trainProfile", "addPhotosProfile", "faceProfile", "videoProfile"].forEach(id => {
        const select = $("#"+id);
        if (select) select.value = requestedProfileId;
      });
    }
    updateFaceConditionControls();
    return profiles;
  }

  const createProfileForm = $("#createProfileForm");
  const profileDescriptionInput = $("#profileDescription");
  const profileDescriptionCount = $("#profileDescriptionCount");
  function updateProfileDescriptionCount() {
    if (!profileDescriptionCount) return;
    const words = (profileDescriptionInput?.value || "").trim().split(/\s+/).filter(Boolean).length;
    profileDescriptionCount.textContent = words + "/200 words";
    profileDescriptionCount.classList.toggle("over-limit", words > 200);
  }
  profileDescriptionInput?.addEventListener("input", updateProfileDescriptionCount);
  updateProfileDescriptionCount();
  const facePageParams = new URLSearchParams(window.location.search);
  const profileCreatedName = facePageParams.get("profile_created");
  const profileCreationError = facePageParams.get("profile_error");
  if (profileCreatedName) {
    const status = $("#createProfileStatus");
    if (status) status.textContent = "Created “" + profileCreatedName + "” successfully and returned to the Face Models page.";
    facePageParams.delete("profile_created");
    const cleanQuery = facePageParams.toString();
    history.replaceState(null, "", window.location.pathname + (cleanQuery ? "?" + cleanQuery : ""));
  } else if (profileCreationError) {
    const status = $("#createProfileStatus");
    if (status) status.textContent = profileCreationError;
    facePageParams.delete("profile_error");
    const cleanQuery = facePageParams.toString();
    history.replaceState(null, "", window.location.pathname + (cleanQuery ? "?" + cleanQuery : ""));
  }

  createProfileForm?.addEventListener("submit", async event => {
    event.preventDefault();
    const button = $("#createProfileButton");
    const status = $("#createProfileStatus");
    const descriptionWords = (profileDescriptionInput?.value || "").trim().split(/\s+/).filter(Boolean).length;
    if (descriptionWords > 200) {
      if (status) status.textContent = "Keep the face description to 200 words or fewer.";
      return;
    }
    if (button) {button.disabled=true;button.textContent="Creating profile…";}
    if (status) status.textContent="Validating photos and saving face profile…";
    try {
      const data = await readJson(await fetch("/api/face-profiles", {method:"POST", headers:{"Accept":"application/json","X-Requested-With":"XMLHttpRequest"}, body:new FormData(createProfileForm)}));
      createProfileForm.reset();
      updateProfileDescriptionCount();
      await refreshProfiles();
      ["trainProfile","faceProfile","videoProfile","addPhotosProfile"].forEach(id => {const node=$("#"+id);if(node)node.value=data.profile.id;});
      updateFaceConditionControls();
      if (status) status.textContent="Created “"+data.profile.name+"” with "+data.profile.image_count+" photo(s). Prompt token: "+data.profile.token+".";
    } catch (error) { if (status) status.textContent=error.message; }
    finally { if (button) {button.disabled=false;button.textContent="Create profile";} }
  });

  const addPhotosButton = $("#addPhotosButton");
  addPhotosButton?.addEventListener("click", async () => {
    const id = $("#addPhotosProfile")?.value;
    const files = $("#morePhotos")?.files;
    const status = $("#addPhotosStatus");
    const requestedGroupId = new URLSearchParams(window.location.search).get("group_id");
    if (!id) {if(status)status.textContent="Select the profile that should receive these photos.";return;}
    if (!files?.length) {if(status)status.textContent="Choose one or more local photos first.";return;}
    addPhotosButton.disabled=true;
    if(status)status.textContent="Saving selected local photos…";
    const data=new FormData();[...files].forEach(file=>data.append("photos",file));
    if(requestedGroupId)data.set("group_id",requestedGroupId);
    try {
      const result=await readJson(await fetch("/api/face-profiles/"+id+"/photos",{method:"POST",body:data}));
      $("#morePhotos").value="";
      await refreshProfiles();
      await loadDatasetGroups();
      if(status)status.textContent="Added "+result.added+" photo(s) to "+result.profile.name+(result.group_id?" and the selected dataset group.":"." )+" Total: "+result.profile.image_count+".";
    } catch(error) {if(status)status.textContent=error.message;}
    finally {addPhotosButton.disabled=false;}
  });

  function updateGeneratedPhotoSelectionCount() {
    const label=$("#generatedSelectionCount");
    if(label)label.textContent=selectedGeneratedAssetIds.size+" image(s) selected across all pages.";
  }
  async function loadGeneratedPhotoPicker() {
    const picker=$("#generatedPhotoPicker");
    if(!picker) return;
    const previous=$("#generatedPhotoPrev");const next=$("#generatedPhotoNext");
    const pageLabel=$("#generatedPhotoPageLabel");
    const search=$("#generatedPhotoSearch");
    if(previous)previous.disabled=true;if(next)next.disabled=true;
    try {
      const params=new URLSearchParams({
        type:"image",limit:String(generatedPhotoLimit),offset:String(generatedPhotoOffset),
        q:search?.value||"",sort:"newest"
      });
      const data=await readJson(await fetch("/api/assets?"+params.toString()));
      const assets=data.assets||[];
      generatedPhotoTotal=Number(data.total||0);
      picker.replaceChildren();
      const pages=Math.max(1,Math.ceil(generatedPhotoTotal/generatedPhotoLimit));
      if(pageLabel)pageLabel.textContent="Page "+(Math.floor(generatedPhotoOffset/generatedPhotoLimit)+1)+" of "+pages;
      if(previous)previous.disabled=generatedPhotoOffset<=0;
      if(next)next.disabled=generatedPhotoOffset+generatedPhotoLimit>=generatedPhotoTotal;
      if(!assets.length) {
        const empty=document.createElement("div");empty.className="empty-state";
        empty.textContent=(search?.value||"").trim()?"No saved images match that search.":"No generated images saved yet. Generate an image first.";
        picker.appendChild(empty);updateGeneratedPhotoSelectionCount();return;
      }
      assets.forEach(asset=>{
        generatedPhotoAssetCache.set(asset.id,asset);
        const label=document.createElement("label");label.className="generated-photo-option";
        const checkbox=document.createElement("input");checkbox.type="checkbox";checkbox.name="generatedAsset";checkbox.value=asset.id;checkbox.checked=selectedGeneratedAssetIds.has(asset.id);checkbox.setAttribute("aria-label","Select "+asset.title);
        checkbox.addEventListener("change",()=>{
          if(checkbox.checked)selectedGeneratedAssetIds.add(asset.id);
          else selectedGeneratedAssetIds.delete(asset.id);
          updateGeneratedPhotoSelectionCount();
        });
        const image=document.createElement("img");image.src="/assets/"+asset.id+"/content";image.alt=asset.title;image.loading="lazy";
        const title=document.createElement("span");title.textContent=asset.title;
        label.append(checkbox,image,title);picker.appendChild(label);
      });
      updateGeneratedPhotoSelectionCount();
    } catch(error) {
      picker.replaceChildren();
      const empty=document.createElement("div");empty.className="empty-state";empty.textContent="Could not load images: "+error.message;picker.appendChild(empty);
    }
  }
  let selectedVariationSourceAssetId="";
  $("#useGeneratedAsVariationSource")?.addEventListener("click",()=>{
    const selected=[...selectedGeneratedAssetIds];
    const status=$("#generatedPhotosStatus");
    if(selected.length!==1){if(status)status.textContent="Select exactly one generated image to use as the variation source.";return;}
    const asset=generatedPhotoAssetCache.get(selected[0]);
    if(!asset){if(status)status.textContent="That image is not loaded yet. Refresh the gallery and select it again.";return;}
    selectedVariationSourceAssetId=asset.id;
    const preview=$("#variationSourcePreview");
    const previewImage=$("#variationSourcePreviewImage");
    const previewText=$("#variationSourcePreviewText");
    if(previewImage)previewImage.src="/assets/"+asset.id+"/content";
    if(previewText)previewText.textContent="Using saved image: "+asset.title;
    if(preview)preview.hidden=false;
    const localSource=$("#angleSourceImage");if(localSource)localSource.value="";
    const anglesTab=$('.source-tab[data-source-tab="'+(activeVariationMode==="expressions"?"expressions":"angles")+'"]');
    if(anglesTab)activateTrainingSourceTab(anglesTab);
    if(status)status.textContent="Saved image selected as the source for the next angle/expression batch.";
  });
  $("#clearVariationSource")?.addEventListener("click",()=>{
    selectedVariationSourceAssetId="";
    const preview=$("#variationSourcePreview");if(preview)preview.hidden=true;
  });
  $("#angleSourceImage")?.addEventListener("change",()=>{
    if($("#angleSourceImage")?.files?.length){
      selectedVariationSourceAssetId="";
      const preview=$("#variationSourcePreview");if(preview)preview.hidden=true;
    }
  });
  $("#generatedPhotoSearch")?.addEventListener("input",()=>{generatedPhotoOffset=0;loadGeneratedPhotoPicker();});
  $("#generatedPhotoPrev")?.addEventListener("click",()=>{generatedPhotoOffset=Math.max(0,generatedPhotoOffset-generatedPhotoLimit);loadGeneratedPhotoPicker();});
  $("#generatedPhotoNext")?.addEventListener("click",()=>{if(generatedPhotoOffset+generatedPhotoLimit<generatedPhotoTotal){generatedPhotoOffset+=generatedPhotoLimit;loadGeneratedPhotoPicker();}});
  $("#addGeneratedPhotosButton")?.addEventListener("click",async()=>{
    const id=$("#addPhotosProfile")?.value;
    const selected=[...selectedGeneratedAssetIds];
    const requestedGroupId=new URLSearchParams(window.location.search).get("group_id");
    const status=$("#generatedPhotosStatus");const button=$("#addGeneratedPhotosButton");
    if(!id){if(status)status.textContent="Select the face profile that should receive the images.";return;}
    if(!selected.length){if(status)status.textContent="Select at least one generated image.";return;}
    if(selected.length>30){if(status)status.textContent="Select at most 30 generated images per batch.";return;}
    button.disabled=true;if(status)status.textContent="Copying selected database images into the face dataset…";
    try{
      const angleLabels=selected.map((assetId,index)=>{
        const asset=generatedPhotoAssetCache.get(assetId);
        return String(asset?.title||("Additional photo "+(index+1))).slice(0,80);
      });
      const payload={asset_ids:selected};
      if(requestedGroupId){payload.group_id=requestedGroupId;payload.angle_labels=angleLabels;}
      const result=await readJson(await fetch("/api/face-profiles/"+id+"/photos/from-assets",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}));
      selected.forEach(assetId=>selectedGeneratedAssetIds.delete(assetId));
      await refreshProfiles();
      await loadGeneratedPhotoPicker();
      await loadDatasetGroups();
      if(status)status.textContent="Added "+result.added+" generated image(s) to "+result.profile.name+(requestedGroupId?" and the selected dataset group.":"." )+" Total: "+result.profile.image_count+".";

    }catch(error){if(status)status.textContent=error.message;}
    finally{button.disabled=false;updateGeneratedPhotoSelectionCount();}
  });

  const generateFaceAnglesButton=$("#generateFaceAnglesButton");
  const retryFaceAngleButton=$("#retryFaceAngleButton");
  let activeVariationMode="face_angles";
  const faceVariationLabels={
    face_angles:["Front view","Middle-left","Middle-right","Three-quarter left","Three-quarter right","Left profile","Right profile","Looking up","Looking down"],
    expressions:["Neutral expression","Gentle smile","Broad smile","Laughing","Sad","Angry","Surprised","Fearful","Disgusted","Confused"]
  };
  // Keep each mode's target selection independent when switching between angles and expressions.
  const selectedFaceTargetsByMode={
    face_angles:new Set(faceVariationLabels.face_angles),
    expressions:new Set(faceVariationLabels.expressions)
  };
  const faceTargetChecklist=$("#faceTargetChecklist");
  const selectAllFaceTargets=$("#selectAllFaceTargets");
  function selectedFaceTargetLabels(){
    return Array.from((faceTargetChecklist||document).querySelectorAll(".face-target-option input[type=checkbox]:checked"), input => input.value);
  }
  function renderFaceTargetChecklist(){
    if(!faceTargetChecklist)return;
    const labels=faceVariationLabels[activeVariationMode]||[];
    const selected=selectedFaceTargetsByMode[activeVariationMode]||new Set(labels);
    faceTargetChecklist.innerHTML="";
    labels.forEach((label,index)=>{
      const item=document.createElement("label");
      item.className="face-target-option";
      const checkbox=document.createElement("input");
      checkbox.type="checkbox";checkbox.value=label;checkbox.checked=selected.has(label);
      checkbox.setAttribute("aria-label","Generate "+label);
      const text=document.createElement("span");
      text.textContent=label;
      item.append(checkbox,text);
      faceTargetChecklist.append(item);
      checkbox.addEventListener("change",syncFaceTargetSelection);
    });
    if(selectAllFaceTargets)selectAllFaceTargets.checked=true;
    syncFaceTargetSelection();
  }
  function syncFaceTargetSelection(){
    if(!faceTargetChecklist)return;
    const boxes=Array.from(faceTargetChecklist.querySelectorAll(".face-target-option input"));
    const checked=boxes.filter(box=>box.checked);
    selectedFaceTargetsByMode[activeVariationMode]=new Set(checked.map(box=>box.value));
    if(selectAllFaceTargets){selectAllFaceTargets.checked=boxes.length>0&&checked.length===boxes.length;selectAllFaceTargets.indeterminate=checked.length>0&&checked.length<boxes.length;}
    const help=$("#targetSelectionHelp");
    if(help)help.textContent=checked.length===boxes.length
      ?"All "+boxes.length+" "+(activeVariationMode==="expressions"?"expressions":"face angles")+" are selected."
      :checked.length+" of "+boxes.length+" selected. Only checked targets will be generated.";
    if(generateFaceAnglesButton&&!activeFaceDatasetJobs.some(job=>["queued","running","cancelling"].includes(job.status))&&!startingFaceDatasetJob)
      generateFaceAnglesButton.textContent="Generate "+checked.length+" selected "+(activeVariationMode==="expressions"?"expressions":"face angles");
  }
  selectAllFaceTargets?.addEventListener("change",()=>{
    $(".face-target-option input",faceTargetChecklist||document).forEach(box=>box.checked=selectAllFaceTargets.checked);
    syncFaceTargetSelection();
  });
  let activeFaceDatasetJobs=[];
  let trackedFaceDatasetJobId=sessionStorage.getItem("privateFaceStudioActiveDatasetJob")||"";
  let lastFaceDatasetTerminalJob=null;
  let terminalFaceDatasetVisibleUntil=0;
  let startingFaceDatasetJob=false;
  let lastDatasetRefreshAt=0;
  let backgroundJobPollInFlight=false;

  function activateTrainingSourceTab(tab) {
    $$(".source-tab").forEach(item=>item.classList.toggle("active",item===tab));
    const key=tab.dataset.sourceTab;
    if(key==="angles"||key==="expressions")activeVariationMode=key==="expressions"?"expressions":"face_angles";
    $$("[data-source-panel]").forEach(panel=>{
      panel.hidden=panel.dataset.sourcePanel==="variations"
        ? !(key==="angles"||key==="expressions")
        : panel.dataset.sourcePanel!==key;
    });
    const generateButton=$("#generateFaceAnglesButton");
    const description=$("#variationModeDescription");
    const title=$("#angleGroupTitle");
    if(generateButton)generateButton.textContent=activeVariationMode==="expressions"
      ?"Start 10 facial expressions":"Start 9 face angles";
    if(description)description.textContent=activeVariationMode==="expressions"
      ?"Create ten separately labeled facial expressions while preserving the source head angle. Results are pose-checked and saved one by one."
      :"Create nine explicit head-pose targets: front, middle-left/right, three-quarter left/right, left/right profiles, looking up, and looking down. Every result is pose-checked and saved one by one.";
    if(title&&!(title.value||"").trim())title.placeholder=activeVariationMode==="expressions"
      ?"e.g. Model A · expression set":"e.g. Model A · nine-angle set";
    if(retryFaceAngleButton)retryFaceAngleButton.hidden=true;
    renderFaceTargetChecklist();
  }
  // Guard this optional page section so a missing/null selector result cannot abort all studio initialization.
  // $() returns only the first matching element; use $() for all source tabs.
  // This page has multiple source tabs. Use $$ (querySelectorAll), not $ (querySelector):
  // $ returns one element, so calling forEach on it throws and prevents every handler below
  // from registering, including Generate photo and strength-slider synchronization.
  const sourceTabs = $$(".source-tab");
  sourceTabs.forEach(tab => tab.addEventListener("click", () => activateTrainingSourceTab(tab)));
  const initialTrainingTab=$('.source-tab[data-source-tab="angles"]');
  if(initialTrainingTab)activateTrainingSourceTab(initialTrainingTab);

  function trainingJobLabel(job) {
    return job.dataset_type==="expressions"?"Face expressions":"Face angles";
  }
  function renderBackgroundFaceDatasetJob(job, {terminal=false}={}) {
    const banner=$("#backgroundTrainingJobBanner");
    if(!banner)return;
    banner.hidden=false;
    const labels=faceVariationLabels[job.dataset_type]||[];
    const expected=Number(job.expected_count||labels.length||1);
    const done=Number(job.generated_count||0);
    const targetName=job.current_label||labels[Math.max(0,Number(job.current_index||1)-1)]||"Preparing";
    const sub=job.generation_progress;
    const subProgress=sub&&["running","cancelling"].includes(sub.status)
      ?Math.max(0,Math.min(99,Number(sub.progress||0))):0;
    const percent=terminal
      ?(job.status==="completed"?100:Math.max(0,Math.min(100,Math.round(100*done/Math.max(1,expected)))))
      :Math.max(0,Math.min(100,Math.round(100*(done+subProgress/100)/Math.max(1,expected))));
    const title=$("#backgroundTrainingJobTitle");
    const state=$("#backgroundTrainingJobState");
    const status=$("#backgroundTrainingJobStatus");
    const progress=$("#backgroundTrainingJobProgress");
    const counts=$("#backgroundTrainingJobCounts");
    const cancel=$("#cancelBackgroundTrainingJob");
    const retry=$("#retryFailedBackgroundTrainingJob");
    if(title)title.textContent=(job.title||trainingJobLabel(job))+" · "+trainingJobLabel(job);
    if(state){
      state.textContent=job.status==="cancelling"?"Cancelling…":job.status==="queued"?"Queued":job.status==="running"?"Running":job.status==="completed"?"Completed":job.status==="cancelled"?"Cancelled":job.status==="interrupted"?"Interrupted":job.status==="failed"?"Failed":"Partial";
      state.className="dataset-group-status "+(job.status==="completed"?"completed":["failed","partial","cancelled","interrupted"].includes(job.status)?"failed":"generating");
    }
    if(status){
      if(job.status==="completed")status.textContent="All targets saved. "+(job.review_targets?.length||0)+" image(s) need manual pose review.";
      else if(job.status==="cancelled")status.textContent="Cancelled. Previously saved images were preserved.";
      else if(job.status==="failed")status.textContent=job.error||"The generation job failed before it could complete.";
      else if(job.status==="partial"||job.status==="interrupted")status.textContent=job.error||"Some targets did not finish; saved images remain available.";
      else if(job.status==="cancelling")status.textContent="Cancellation requested. The current denoising step will stop when the backend reaches its cancellation check.";
      else status.textContent=(sub?.message||("Generating "+targetName+"…"))+" "+targetName+" ("+Math.min(Number(job.current_index||1),expected)+"/"+expected+").";
    }
    if(progress){progress.value=percent;progress.setAttribute("aria-valuenow",String(percent));}
    if(counts)counts.textContent=done+" of "+expected+" targets saved · "+(job.failed_targets?.length||0)+" failed · "+(job.review_targets?.length||0)+" need review";
    if(cancel){
      cancel.hidden=terminal||!["queued","running","cancelling"].includes(job.status);
      cancel.disabled=job.status==="cancelling";
      cancel.textContent=job.status==="cancelling"?"Cancelling…":"Cancel generation";
    }
    if(retry){
      retry.hidden=!terminal||!(job.failed_targets||[]).length;
      retry.disabled=false;
      retry.textContent="Retry "+(job.failed_targets||[]).length+" failed target(s)";
    }
  }
  function syncFaceDatasetStartButton() {
    if(!generateFaceAnglesButton)return;
    const running=activeFaceDatasetJobs.some(job=>["queued","running","cancelling"].includes(job.status));
    generateFaceAnglesButton.disabled=startingFaceDatasetJob||running;
    if(!running&&!startingFaceDatasetJob)generateFaceAnglesButton.textContent=activeVariationMode==="expressions"?"Start 10 facial expressions":"Start 9 face angles";
    else if(startingFaceDatasetJob)generateFaceAnglesButton.textContent="Starting background job…";
    if(retryFaceAngleButton)retryFaceAngleButton.hidden=true;
  }
  async function getFaceDatasetJob(jobId) {
    return readJson(await fetch("/api/training-generation-jobs/"+encodeURIComponent(jobId),{cache:"no-store"}));
  }
  async function refreshBackgroundFaceDatasetJobs() {
    if(backgroundJobPollInFlight)return;
    backgroundJobPollInFlight=true;
    try{
      const data=await readJson(await fetch("/api/training-generation-jobs/active",{cache:"no-store"}));
      activeFaceDatasetJobs=data.jobs||[];
      const active=activeFaceDatasetJobs[0];
      if(active){
        trackedFaceDatasetJobId=active.id;
        sessionStorage.setItem("privateFaceStudioActiveDatasetJob",active.id);
        // Fetch fresh per-denoising-step progress without tying the work to this page.
        if(active.current_task_id){
          try{
            const progress=await readJson(await fetch("/api/generation-progress/"+encodeURIComponent(active.current_task_id),{cache:"no-store"}));
            active.generation_progress=progress;
          }catch(_){}
        }
        renderBackgroundFaceDatasetJob(active);
        if($("#faceAnglesStatus")){
          $("#faceAnglesStatus").textContent=(active.status==="cancelling"?"Cancelling ":active.status==="queued"?"Queued ":"Generating ")+
            trainingJobLabel(active)+" · "+(active.current_label||"preparing")+ " · "+active.generated_count+"/"+active.expected_count+" saved.";
        }
        if($("#faceAnglesProgress")){
          $("#faceAnglesProgress").hidden=false;
          const bar=$("#faceAnglesProgressBar");const percent=$("#faceAnglesProgressPercent");
          const progress=active.generation_progress;
          const p=Math.round(100*(Number(active.generated_count||0)+(progress?.progress||0)/100)/Math.max(1,Number(active.expected_count||1)));
          if(bar)bar.value=Math.min(100,p);
          if(percent)percent.textContent=Math.min(100,p)+"%";
          const stage=$("#faceAnglesProgressStage");const detail=$("#faceAnglesProgressDetail");
          if(stage)stage.textContent=active.status==="cancelling"?"Cancelling":trainingJobLabel(active)+" · "+(active.current_label||"starting");
          if(detail)detail.textContent=(progress?.message||"Background job is running on the server.")+" Completed "+active.generated_count+" of "+active.expected_count+".";
        }
        if($("#datasetGroups")&&Date.now()-lastDatasetRefreshAt>1800){
          lastDatasetRefreshAt=Date.now();
          loadDatasetGroups();
        }
      }else{
        activeFaceDatasetJobs=[];
        syncFaceDatasetStartButton();
        if(trackedFaceDatasetJobId){
          try{
            const response=await getFaceDatasetJob(trackedFaceDatasetJobId);
            const finished=response.job;
            if(!["queued","running","cancelling"].includes(finished.status)){
              lastFaceDatasetTerminalJob=finished;
              terminalFaceDatasetVisibleUntil=Date.now()+((finished.failed_targets||[]).length?24*60*60*1000:12000);
              renderBackgroundFaceDatasetJob(finished,{terminal:true});
              sessionStorage.removeItem("privateFaceStudioActiveDatasetJob");
              if($("#faceAnglesStatus"))$("#faceAnglesStatus").textContent=(finished.error||(
                finished.status==="completed"?"Generation completed. Review every image marked Needs review.":"Generation "+finished.status+". Completed images were preserved."
              ));
              if($("#faceAnglesProgress")){
                const bar=$("#faceAnglesProgressBar");const percent=$("#faceAnglesProgressPercent");
                if(bar)bar.value=100;if(percent)percent.textContent="100%";
                const stage=$("#faceAnglesProgressStage");const detail=$("#faceAnglesProgressDetail");
                if(stage)stage.textContent=finished.status==="completed"?"Dataset ready":("Dataset "+finished.status);
                if(detail)detail.textContent=(finished.error||"All saved images remain in the dataset for review.")+" "+(finished.review_targets?.length||0)+" needs review.";
              }
              trackedFaceDatasetJobId="";
              if($("#datasetGroups"))loadDatasetGroups();
            }
          }catch(_){
            sessionStorage.removeItem("privateFaceStudioActiveDatasetJob");
            trackedFaceDatasetJobId="";
          }
        }else if(lastFaceDatasetTerminalJob&&(
          Date.now()<terminalFaceDatasetVisibleUntil||(lastFaceDatasetTerminalJob.failed_targets||[]).length
        )){
          renderBackgroundFaceDatasetJob(lastFaceDatasetTerminalJob,{terminal:true});
        }else{
          const banner=$("#backgroundTrainingJobBanner");
          if(banner)banner.hidden=true;
        }
      }
    }catch(error){
      // Keep any existing status visible. Polling failure must never interrupt the server job.
      const status=$("#backgroundTrainingJobStatus");
      if(status&&!activeFaceDatasetJobs.length)status.textContent="Background job status temporarily unavailable. Generation continues on the server.";
    }finally{
      backgroundJobPollInFlight=false;
      syncFaceDatasetStartButton();
    }
  }

  generateFaceAnglesButton?.addEventListener("click",async()=>{
    const profileId=$("#addPhotosProfile")?.value;
    const source=$("#angleSourceImage")?.files?.[0];
    const titleInput=$("#angleGroupTitle");
    const variationSourceAssetId=selectedVariationSourceAssetId;
    const title=(titleInput?.value||"").trim()||((activeVariationMode==="expressions"?"Face expressions · ":"Face angles · ")+new Date().toLocaleString());
    const status=$("#faceAnglesStatus");
    if(!profileId){if(status)status.textContent="Select the face profile that should receive these training photos.";return;}
    if(!source&&!variationSourceAssetId){if(status)status.textContent="Choose a local source photo, or select a saved image in From generated images and use it as the source.";return;}
    const selectedTargets=selectedFaceTargetLabels();
    if(!selectedTargets.length){if(status)status.textContent="Select at least one angle or expression to generate.";return;}
    if(source&&source.size>8*1024*1024){if(status)status.textContent="The source image must be 8 MB or smaller.";return;}
    if(activeFaceDatasetJobs.some(job=>["queued","running","cancelling"].includes(job.status))){if(status)status.textContent="A batch is already running. Cancel it or wait for it to finish.";return;}
    startingFaceDatasetJob=true;syncFaceDatasetStartButton();
    const localProgress=$("#faceAnglesProgress");
    const localProgressBar=$("#faceAnglesProgressBar");
    const localProgressPercent=$("#faceAnglesProgressPercent");
    const localProgressStage=$("#faceAnglesProgressStage");
    const localProgressDetail=$("#faceAnglesProgressDetail");
    if(localProgress)localProgress.hidden=false;
    if(localProgressBar)localProgressBar.value=0;
    if(localProgressPercent)localProgressPercent.textContent="0%";
    if(localProgressStage)localProgressStage.textContent=source?"Uploading source photo":"Loading saved source image";
    if(localProgressDetail)localProgressDetail.textContent="Saving the source photo and registering a server-managed job. Image generation has not been confirmed yet.";
    if(status)status.textContent="Preparing source photo and creating the background job…";
    try{
      const formData=new FormData();
      formData.set("title",title);
      formData.set("dataset_type",activeVariationMode);
      formData.set("selected_targets",JSON.stringify(selectedTargets));
      if(source)formData.append("source_image",source,source.name);
      else formData.set("source_asset_id",variationSourceAssetId);
      addComputeData(formData);
      const response=await fetch("/api/face-profiles/"+encodeURIComponent(profileId)+"/dataset-jobs",{method:"POST",body:formData});
      const created=await readJson(response);
      if(!created.job||!created.job.id)throw new Error("The server response did not include a background job ID.");
      trackedFaceDatasetJobId=created.job.id;
      sessionStorage.setItem("privateFaceStudioActiveDatasetJob",created.job.id);
      activeFaceDatasetJobs=[created.job];
      lastFaceDatasetTerminalJob=null;
      if(titleInput)titleInput.value=title;
      renderBackgroundFaceDatasetJob(created.job);
      if(status)status.textContent="Background job queued. You can change pages; completed images will be saved one at a time.";
      if(localProgressStage)localProgressStage.textContent="Background job queued";
      if(localProgressDetail)localProgressDetail.textContent="Job ID: "+created.job.id+". Waiting for the worker to process the first target.";
      await refreshBackgroundFaceDatasetJobs();
    }catch(error){
      if(status)status.textContent="Could not start background generation: "+error.message;
      if(localProgressStage)localProgressStage.textContent="Job start failed";
      if(localProgressDetail)localProgressDetail.textContent=error.message+" Check the Flask terminal for the server-side traceback.";
      if(localProgressPercent)localProgressPercent.textContent="—";
      const banner=$("#backgroundTrainingJobBanner");
      if(banner)banner.hidden=true;
      sessionStorage.removeItem("privateFaceStudioActiveDatasetJob");
      trackedFaceDatasetJobId="";
      activeFaceDatasetJobs=[];
    }finally{
      startingFaceDatasetJob=false;
      syncFaceDatasetStartButton();
    }
  });

  $("#cancelBackgroundTrainingJob")?.addEventListener("click",async()=>{
    const job=activeFaceDatasetJobs[0];
    const jobId=job?.id||trackedFaceDatasetJobId;
    const status=$("#backgroundTrainingJobStatus");
    const button=$("#cancelBackgroundTrainingJob");
    if(!jobId)return;
    button.disabled=true;
    if(status)status.textContent="Sending cancellation request to the generation worker…";
    try{
      await readJson(await fetch("/api/training-generation-jobs/"+encodeURIComponent(jobId)+"/cancel",{method:"POST"}));
      if(status)status.textContent="Cancellation requested. Completed images are preserved; the current step is being stopped.";
      await refreshBackgroundFaceDatasetJobs();
    }catch(error){if(status)status.textContent=error.message;}
    finally{button.disabled=false;}
  });

  $("#retryFailedBackgroundTrainingJob")?.addEventListener("click",async()=>{
    const job=lastFaceDatasetTerminalJob;
    const status=$("#backgroundTrainingJobStatus");
    const button=$("#retryFailedBackgroundTrainingJob");
    if(!job?.id||!(job.failed_targets||[]).length)return;
    button.disabled=true;
    if(status)status.textContent="Queueing only failed targets. Images already saved will not be regenerated.";
    try{
      const result=await readJson(await fetch(
        "/api/training-generation-jobs/"+encodeURIComponent(job.id)+"/retry-failed",
        {method:"POST"}
      ));
      trackedFaceDatasetJobId=result.job.id;
      sessionStorage.setItem("privateFaceStudioActiveDatasetJob",result.job.id);
      lastFaceDatasetTerminalJob=null;
      activeFaceDatasetJobs=[result.job];
      if(status)status.textContent=result.message||"Retry queued.";
      await refreshBackgroundFaceDatasetJobs();
    }catch(error){if(status)status.textContent=error.message;}
    finally{button.disabled=false;}
  });

  refreshBackgroundFaceDatasetJobs();
  window.setInterval(refreshBackgroundFaceDatasetJobs,1800);
  let globalBodyStatusCache=null;
  async function loadGlobalBodyStatus(showError=true) {
    const bodyStatus=$("#globalBodyModelStatus");
    const generateStatus=$("#globalBodyGenerationStatus");
    const editBodyCheckbox=$("#trainingPhotoUseBodyModel");
    try {
      globalBodyStatusCache=await readJson(await fetch("/api/global-body/status",{cache:"no-store"}));
      const info=globalBodyStatusCache;
      if(bodyStatus)bodyStatus.textContent="Shared body model · "+info.image_count+" training photo(s) · local: "+(info.local_model_ready?"trained":"not trained")+" · GHive / Colab GPU: "+(info.runpod_model_ready?"trained":"not trained")+". Train after adding at least five samples."; 
      if(generateStatus)generateStatus.textContent="Shared body model status — local: "+(info.local_model_ready?"ready":"not trained")+", remote GPU: "+(info.runpod_model_ready?"ready":"not trained")+". Enable the option only after training on the selected backend.";
      const requestedBackend=useGpuAcceleration?.checked?(computeBackend?.value||"auto"):"cpu";
      const selectedBackend=requestedBackend==="auto"?(computeAvailability?.auto_backend||"auto"):requestedBackend;
      const backendReady=(selectedBackend==="runpod"||selectedBackend==="cloud_gpu")?info.runpod_model_ready:(selectedBackend==="auto"?false:info.local_model_ready);
      const generationBodyCheckbox=$("#useGlobalBodyModel");
      if(generationBodyCheckbox){generationBodyCheckbox.disabled=!backendReady;if(!backendReady)generationBodyCheckbox.checked=false;}
      if(editBodyCheckbox){editBodyCheckbox.disabled=!backendReady;if(editBodyCheckbox.disabled)editBodyCheckbox.checked=false;}
      const trainBodyButton=$("#trainGlobalBodyButton");
      if(trainBodyButton)trainBodyButton.disabled=Number(info.image_count||0)<5;
      return info;
    } catch(error) {
      if(showError&&bodyStatus)bodyStatus.textContent="Could not read shared body-model status: "+error.message;
      if(generateStatus)generateStatus.textContent="Could not read shared body-model status. Check the database connection."; 
      return null;
    }
  }
  $("#trainingPhotoUseBodyModel")?.addEventListener("change",event=>{event.target.dataset.userTouched="1";});
  useGpuAcceleration?.addEventListener("change",()=>loadGlobalBodyStatus(false));
  computeBackend?.addEventListener("change",()=>loadGlobalBodyStatus(false));

  const generateBodyDatasetButton=$("#generateBodyDatasetButton");
  generateBodyDatasetButton?.addEventListener("click",async()=>{
    const titleInput=$("#bodyGroupTitle");
    const promptInput=$("#bodyTypePrompt");
    const status=$("#bodyDatasetStatus");
    const title=(titleInput?.value||"").trim()||("Shared body type · "+new Date().toLocaleString());
    const bodyPrompt=(promptInput?.value||"").trim();
    if(!bodyPrompt){if(status)status.textContent="Describe the adult body type or figure before generating samples.";return;}
    if(bodyPrompt.length>1000){if(status)status.textContent="The body prompt must be 1000 characters or fewer.";return;}
    generateBodyDatasetButton.disabled=true;
    const originalText=generateBodyDatasetButton.textContent;
    generateBodyDatasetButton.textContent="Generating body samples…";
    let group=null,generatedCount=0;
    try {
      group=(await readJson(await fetch("/api/global-body/dataset-groups",{
        method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({title,prompt:bodyPrompt})
      }))).group;
      const info=await loadGlobalBodyStatus();
      if(!info?.profile_id)throw new Error("Could not load the shared body profile.");
      for(let index=0;index<5;index++){
        if(status)status.textContent="Generating body sample "+(index+1)+" of 5. No source photo is used."; 
        const taskId=newGenerationTaskId();
        const formData=new FormData();
        formData.set("prompt","Photorealistic full-body studio photograph of an adult model, "+bodyPrompt+". Entire adult subject visible from head to feet, natural anatomy and realistic proportions, neutral uncluttered studio background, realistic skin texture and soft photographic lighting. Sample "+(index+1)+" of the same requested body type, with a subtly different natural standing pose.");
        formData.set("negative_prompt","cartoon, anime, illustration, CGI, 3D render, plastic skin, deformed anatomy, extra limbs, cropped head, cropped feet, out of frame, blurry, watermark, text");
        formData.set("steps","20");formData.set("title",title+" - Body sample "+(index+1));
        formData.set("image_preset","full_body_photo");formData.set("aspect_ratio","portrait");formData.set("task_id",taskId);
        formData.delete("use_global_body_model");
        addComputeData(formData);
        const controller=new AbortController();
        const watcher=watchGenerationProgress(taskId,"image",controller);
        let generated;
        try {generated=await readJson(await fetch("/generate",{method:"POST",body:formData}));}
        finally {controller.abort();await watcher;}
        if(!generated.asset_id)throw new Error("Body sample "+(index+1)+" was generated without a saved image ID.");
        const label=("Body type: "+bodyPrompt).slice(0,80);
        await readJson(await fetch("/api/face-profiles/"+info.profile_id+"/photos/from-assets",{
          method:"POST",headers:{"Content-Type":"application/json"},
          body:JSON.stringify({asset_ids:[generated.asset_id],group_id:group.id,angle_labels:[label]})
        }));
        generatedCount=index+1;
        await updateFaceDatasetGroup(group.id,{status:"generating",generated_count:generatedCount,error:""});
        if(status)status.textContent="Saved body sample "+generatedCount+" of 5 to the shared dataset. The generated image also remains in your image library."; 
      }
      await updateFaceDatasetGroup(group.id,{status:"completed",generated_count:5,error:""});
      if(status)status.textContent="Created shared body-type dataset “"+title+"” with five prompt-generated samples. Train the shared body model to reuse it with any face.";
      if(titleInput)titleInput.value=title;
      activeDatasetType="body_type";
      $$(".dataset-category-tab").forEach(tab=>tab.classList.toggle("active",tab.dataset.datasetFilter===activeDatasetType));
      await loadDatasetGroups();
      await loadGlobalBodyStatus();
    } catch(error) {
      if(group)await updateFaceDatasetGroup(group.id,{status:generatedCount?"partial":"failed",generated_count:generatedCount,error:error.message}).catch(()=>{});
      if(status)status.textContent="Body dataset generation stopped after "+generatedCount+" saved sample(s): "+error.message;
      await loadDatasetGroups();
    } finally {
      generateBodyDatasetButton.disabled=false;
      generateBodyDatasetButton.textContent=originalText;
      await loadGlobalBodyStatus(false);
    }
  });

  const trainGlobalBodyButton=$("#trainGlobalBodyButton");
  trainGlobalBodyButton?.addEventListener("click",async()=>{
    const status=$("#bodyDatasetStatus")||$("#globalBodyModelStatus");
    const button=trainGlobalBodyButton;
    const info=await loadGlobalBodyStatus();
    if(!info||!info.profile_id){if(status)status.textContent="Could not load the shared body profile.";return;}
    if(Number(info.image_count||0)<5){if(status)status.textContent="Generate or add at least five body-type photos first.";return;}
    const formData=new FormData();formData.set("steps","800");formData.set("resolution","512");addComputeData(formData);
    const requestedBackend=useGpuAcceleration?.checked?(computeBackend?.value||"auto"):"cpu";
    const selectedBackend=requestedBackend==="auto"?(computeAvailability?.auto_backend||"auto"):requestedBackend;
    const backend=(selectedBackend==="runpod"||selectedBackend==="cloud_gpu")?"runpod":(selectedBackend==="local_gpu"?"local":"cpu");
    button.disabled=true;
    try {
      const response=await readJson(await fetch("/api/face-profiles/"+info.profile_id+"/train",{method:"POST",body:formData}));
      if(status)status.textContent=response.message||"Shared body-model training started."; 
      const selected=(response.backend||backend)==="runpod"?"runpod":"local";
      const started=Date.now();
      while(Date.now()-started<8*60*60*1000){
        await new Promise(resolve=>setTimeout(resolve,4000));
        const state=await readJson(await fetch("/api/face-profiles/"+info.profile_id+"/train-status?backend="+selected));
        if(state.status==="completed"){if(status)status.textContent="Shared body model training completed. Enable it during image generation to reuse the body-type conditioning with any face.";break;}
        if(state.status==="failed"){throw new Error(state.error||"Shared body-model training failed.");}
        if(status)status.textContent="Shared body model training is "+state.status+"…";
      }
      await loadGlobalBodyStatus();
    } catch(error) {if(status)status.textContent="Shared body-model training error: "+error.message;}
    finally {button.disabled=Number((globalBodyStatusCache||{}).image_count||0)<5;await loadGlobalBodyStatus(false);}
  });

  const trainForm=$("#trainForm");
  async function pollTraining(profileId, backend) {
    const started=Date.now();let previous="";
    while(Date.now()-started<8*60*60*1000){
      await new Promise(resolve=>setTimeout(resolve,4000));
      try{
        const response=await fetch("/api/face-profiles/"+profileId+"/train-status?backend="+backend);
        const data=await response.json();
        if(!response.ok&&response.status!==502)throw new Error(data.error||"Could not check training status.");
        const state=data.status||"running";
        const status=$("#trainStatus");
        if(state!==previous&&status){status.textContent="Training status: "+state+(data.error?"\n"+data.error:"");previous=state;}
        if(state==="completed"){if(status)status.textContent="Training completed successfully.";await refreshProfiles();return;}
        if(state==="failed"){if(status)status.textContent="Training failed.\n"+(data.error||data.profile?.local_training_error||data.profile?.runpod_training_error||"No error details returned.");await refreshProfiles();return;}
      }catch(error){const status=$("#trainStatus");if(status)status.textContent="Training may still be running; status polling had an issue: "+error.message;}
    }
    if($("#trainStatus"))$("#trainStatus").textContent="Stopped polling after 8 hours. Check the worker logs and profile state.";
  }
  trainForm?.addEventListener("submit",async event=>{
    event.preventDefault();const id=$("#trainProfile")?.value;
    const status=$("#trainStatus");const button=$("#trainButton");
    const profile=profiles.find(item=>item.id===id);
    if(!id){if(status)status.textContent="Create or select a face profile first.";return;}
    if(!profile||profile.image_count<5){if(status)status.textContent="Add at least 5 good photos to this profile before training.";return;}
    button.disabled=true;if(status)status.textContent="Checking selected compute backend and submitting the face training job…";
    const backend=useGpuAcceleration?.checked?(computeBackend?.value||"auto"):"cpu";
    try{
      const data=await readJson(await fetch("/api/face-profiles/"+id+"/train",{method:"POST",body:addComputeData(new FormData(trainForm))}));
      if(status)status.textContent=data.message||"Training started.";
      await refreshProfiles();
      const resolvedBackend=data.backend||backend;
      await pollTraining(id,(resolvedBackend==="runpod"||resolvedBackend==="cloud_gpu")?"runpod":"local");
    }catch(error){if(status)status.textContent=error.message;}
    finally{button.disabled=false;await refreshProfiles().catch(()=>{});}
  });

  const referenceImage=$("#referenceImage");
  referenceImage?.addEventListener("change",()=>{
    const file=referenceImage.files[0];const preview=$("#sourcePreview");const placeholder=$("#sourcePlaceholder");
    if(!file){preview.hidden=true;preview.removeAttribute("src");placeholder.hidden=false;return;}
    preview.src=URL.createObjectURL(file);preview.hidden=false;placeholder.hidden=true;
  });
  const strengthSlider = $("#strength");
  const strengthNumber = $("#strengthNumber");
  function syncStrength(source, target) {
    if (!source || !target) return;
    const parsed = Number(source.value);
    if (!Number.isFinite(parsed)) return;
    const value = Math.min(0.95, Math.max(0.20, Math.round(parsed * 100) / 100));
    target.value = value.toFixed(2);
    if (source !== strengthSlider && strengthSlider) strengthSlider.value = String(value);
    if (source !== strengthNumber && strengthNumber) strengthNumber.value = value.toFixed(2);
  }
  strengthSlider?.addEventListener("input", () => syncStrength(strengthSlider, strengthNumber));
  strengthNumber?.addEventListener("input", () => {
    if (strengthNumber.value !== "") syncStrength(strengthNumber, strengthSlider);
  });
  strengthNumber?.addEventListener("change", () => syncStrength(strengthNumber, strengthSlider));

  function newGenerationTaskId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID().replace(/-/g, "").toLowerCase();
    }
    return Array.from({length:32},()=>Math.floor(Math.random()*16).toString(16)).join("");
  }
  function updateGenerationProgress(type, data) {
    const prefix = type === "video" ? "video" : "image";
    const container = $("#"+prefix+"Progress");
    if (!container) return;
    container.hidden = false;
    const percent = Math.max(0, Math.min(100, Number(data.progress || 0)));
    const bar = $("#"+prefix+"ProgressBar");
    const percentLabel = $("#"+prefix+"ProgressPercent");
    const stageLabel = $("#"+prefix+"ProgressStage");
    const detail = $("#"+prefix+"ProgressDetail");
    const step = $("#"+prefix+"ProgressStep");
    if (bar) { bar.value=percent; bar.setAttribute("aria-valuetext",percent+"%"); }
    if (percentLabel) percentLabel.textContent=percent+"%";
    const stageNames = {
      starting:"Getting ready", connecting:"Connecting to GPU", loading_model:"Loading model",
      denoising:"Generating "+(type==="video"?"video":"image"), finalizing:"Finalizing result",
      encoding:"Encoding video", transferring:"Receiving result", saving:"Saving to library", completed:"Complete",
      failed:"Generation failed", busy:"GPU busy"
    };
    if(stageLabel)stageLabel.textContent=stageNames[data.stage]||"Generation in progress";
    if(detail)detail.textContent=data.message||"Working…";
    if(step){
      step.textContent=Number(data.total_steps)>0
        ? "Denoising step "+Number(data.current_step||0)+" of "+Number(data.total_steps)
        : (data.backend==="cloud_gpu"||data.backend==="runpod"?"GHive / Colab GPU":"Local compute");
      if(data.status==="completed")step.textContent="All steps finished · Saved to your library";
      if(data.status==="failed")step.textContent="The task ended before completion.";
    }
    if(data.status==="completed"&&bar)bar.value=100;
    if(data.status==="failed"&&bar)bar.classList.add("progress-failed");
    else if(bar)bar.classList.remove("progress-failed");
  }
  function setFinalGenerationProgress(type, completed, message) {
    updateGenerationProgress(type,{
      status:completed?"completed":"failed",progress:completed?100:0,
      stage:completed?"completed":"failed",message:message|| (completed?"Generation completed.":"Generation failed.")
    });
  }
  async function watchGenerationProgress(taskId, type, controller) {
    while(!controller.signal.aborted){
      try{
        const response=await fetch("/api/generation-progress/"+taskId,{cache:"no-store",signal:controller.signal});
        if(response.ok){
          const data=await response.json();
          updateGenerationProgress(type,data);
          if(data.status==="completed"||data.status==="failed")return;
        }
      }catch(error){
        if(controller.signal.aborted||error.name==="AbortError")return;
      }
      await new Promise(resolve=>setTimeout(resolve,800));
    }
  }

  const generationProfile = $("#generationProfile");
  const strengthControl = $("#strength");
  const stepsControl = $("#steps");
  generationProfile?.addEventListener("change", () => {
    if (generationProfile.value === "preserve_reference") {
      if (strengthControl) strengthControl.value = "0.30";
      if (strengthNumber) strengthNumber.value = "0.30";
      if (stepsControl) stepsControl.value = "25";
    } else if (generationProfile.value === "photorealistic") {
      if (strengthControl) strengthControl.value = "0.45";
      if (strengthNumber) strengthNumber.value = "0.45";
      if (stepsControl) stepsControl.value = "25";
      const preset = $("#imagePreset");
      if (preset) preset.value = "photorealistic";
    }
  });
  const generationForm = $("#generationForm");
  generationForm?.addEventListener("submit",async event=>{
    event.preventDefault();
    const button=$("#generateButton");const status=$("#generationStatus");
    const taskId=newGenerationTaskId();
    const progressController=new AbortController();
    button.disabled=true;button.textContent="Generating image…";
    const progress=$("#imageProgress");if(progress)progress.hidden=false;
    updateGenerationProgress("image",{status:"starting",progress:0,stage:"starting",message:"Submitting image generation request…"});
    if(status)status.textContent=useGpuAcceleration?.checked?"Sending this image task to the selected GPU worker…":"Loading the model and generating locally. CPU inference may be slow.";
    const preview=$("#resultPreview");const placeholder=$("#resultPlaceholder");const download=$("#downloadImage");const pill=$("#imageSavedPill");
    preview.hidden=true;placeholder.hidden=false;download.hidden=true;
    const formData=addComputeData(new FormData(generationForm));formData.set("task_id",taskId);
    const progressWatcher=watchGenerationProgress(taskId,"image",progressController);
    try{
      const data=await readJson(await fetch("/generate",{method:"POST",body:formData}));
      preview.src=data.image_url;preview.hidden=false;placeholder.hidden=true;
      download.href=data.image_url+"?download=1";download.hidden=false;
      pill.textContent="Saved to library";
      setFinalGenerationProgress("image",true,data.message||"Image generated and saved to the library.");
      if(status)status.textContent=(data.message||"Image generated.")+" Saved as “"+data.title+"”.";
    }catch(error){
      setFinalGenerationProgress("image",false,error.message);
      if(status)status.textContent="Generation failed: "+error.message;
    }
    finally{progressController.abort();await progressWatcher;button.disabled=false;button.textContent="Generate photo";}
  });
  const videoForm=$("#videoForm");
  videoForm?.addEventListener("submit",async event=>{
    event.preventDefault();
    const button=$("#videoButton");const status=$("#videoStatus");
    const taskId=newGenerationTaskId();
    const progressController=new AbortController();
    button.disabled=true;button.textContent="Generating video…";
    const progress=$("#videoProgress");if(progress)progress.hidden=false;
    updateGenerationProgress("video",{status:"starting",progress:0,stage:"starting",message:"Submitting video generation request…"});
    if(status)status.textContent=useGpuAcceleration?.checked?"Sending this video task to the selected GPU worker…":"Checking local CUDA and video model files…";
    const preview=$("#videoPreview");const placeholder=$("#videoPlaceholder");const download=$("#downloadVideo");const pill=$("#videoSavedPill");
    preview.hidden=true;preview.removeAttribute("src");placeholder.hidden=false;download.hidden=true;
    const formData=addComputeData(new FormData(videoForm));formData.set("task_id",taskId);
    const progressWatcher=watchGenerationProgress(taskId,"video",progressController);
    try{
      const data=await readJson(await fetch("/generate-video",{method:"POST",body:formData}));
      preview.src=data.video_url;preview.hidden=false;placeholder.hidden=true;
      download.href=data.video_url+"?download=1";download.hidden=false;
      pill.textContent="Saved to library";
      setFinalGenerationProgress("video",true,data.message||"Video generated and saved to the library.");
      if(status)status.textContent=(data.message||"Video generated.")+" Saved as “"+data.title+"”.";
    }catch(error){
      setFinalGenerationProgress("video",false,error.message);
      if(status)status.textContent="Video generation failed: "+error.message;
    }
    finally{progressController.abort();await progressWatcher;button.disabled=false;button.textContent="Generate video";}
  });

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
  }
  const assetGrid=$("#assetGrid");
  const assetType=document.body.dataset.assetType;
  const searchInput=$("#assetSearch");
  const sortInput=$("#assetSort");
  const countLabel=$("#assetCount");
  const libraryStatus=$("#libraryStatus");
  function updateAssetPagination() {
    const pagination=$("#assetPagination");
    const previous=$("#assetPrev");
    const next=$("#assetNext");
    const label=$("#assetPageLabel");
    const pages=Math.max(1,Math.ceil(assetTotal/assetPageSize));
    const current=Math.floor(assetOffset/assetPageSize)+1;
    if(pagination)pagination.hidden=assetTotal<=assetPageSize;
    if(previous)previous.disabled=assetOffset<=0;
    if(next)next.disabled=assetOffset+assetPageSize>=assetTotal;
    if(label)label.textContent="Page "+current+" of "+pages;
  }
  function renderAssets(assets) {
    if(!assetGrid) return;
    const query=(searchInput?.value||"").trim();
    assetGrid.replaceChildren();
    if(countLabel) {
      const first=assetTotal?assetOffset+1:0;
      const last=Math.min(assetOffset+assets.length,assetTotal);
      countLabel.textContent=assetTotal+" saved "+(assetType==="image"?"image(s)":"video(s)")+(assetTotal?" · showing "+first+"–"+last:"");
    }
    updateAssetPagination();
    if(!assets.length){
      const empty=document.createElement("div");empty.className="empty-state panel";
      const title=document.createElement("h3");title.textContent=query?"No matching media":"Nothing saved here yet";
      const detail=document.createElement("p");detail.textContent=query?"Try another search term.":"Create your first "+assetType+" and it will appear in this library.";
      empty.append(title,detail);assetGrid.appendChild(empty);return;
    }
    assets.forEach(asset=>{
      const card=document.createElement("article");card.className="asset-card panel";
      let media;
      if(asset.asset_type==="image"){media=document.createElement("img");media.alt=asset.title;media.loading="lazy";media.src="/assets/"+asset.id+"/content";}
      else {media=document.createElement("video");media.controls=true;media.preload="metadata";media.src="/assets/"+asset.id+"/content";}
      media.className="asset-media";card.appendChild(media);
      const title=document.createElement("h3");title.textContent=asset.title;card.appendChild(title);
      const prompt=document.createElement("p");prompt.className="asset-prompt";prompt.textContent=asset.prompt||"No prompt details saved.";card.appendChild(prompt);
      const meta=document.createElement("div");meta.className="asset-meta";
      [asset.backend||"unknown backend",asset.profile_name||"Base model",asset.created_at?new Date(asset.created_at).toLocaleString(): "Saved",formatBytes(asset.media_size||0)].forEach(text=>{const tag=document.createElement("span");tag.textContent=text;meta.appendChild(tag);});
      card.appendChild(meta);
      const actions=document.createElement("div");actions.className="asset-actions";
      const download=document.createElement("a");download.className="button-link";download.textContent="Download ↓";download.href="/assets/"+asset.id+"/content?download=1";
      const edit=document.createElement("button");edit.type="button";edit.className="secondary-button";edit.textContent="Edit details";edit.addEventListener("click",()=>openEditAsset(asset));
      const remove=document.createElement("button");remove.type="button";remove.className="danger-button";remove.textContent="Delete";remove.addEventListener("click",()=>deleteAsset(asset,remove));
      actions.append(download,edit,remove);card.appendChild(actions);assetGrid.appendChild(card);
    });
  }
  function formatBytes(bytes) {
    if(bytes<1024)return bytes+" B";
    if(bytes<1024*1024)return (bytes/1024).toFixed(1)+" KB";
    if(bytes<1024*1024*1024)return (bytes/1024/1024).toFixed(1)+" MB";
    return (bytes/1024/1024/1024).toFixed(2)+" GB";
  }
  async function loadAssets() {
    if(!assetGrid||!assetType)return;
    try{
      const params=new URLSearchParams({
        type:assetType,limit:String(assetPageSize),offset:String(assetOffset),
        q:searchInput?.value||"",sort:sortInput?.value||"newest"
      });
      const data=await readJson(await fetch("/api/assets?"+params.toString()));
      assetCache.clear();(data.assets||[]).forEach(asset=>assetCache.set(asset.id,asset));
      assetTotal=Number(data.total||0);assetOffset=Number(data.offset||0);
      renderAssets([...assetCache.values()]);
    }catch(error){
      assetGrid.replaceChildren();const empty=document.createElement("div");empty.className="empty-state panel";empty.textContent="Could not load saved media: "+error.message;assetGrid.appendChild(empty);
      if(countLabel)countLabel.textContent="Library unavailable";
    }
  }
  searchInput?.addEventListener("input",()=>{assetOffset=0;loadAssets();});
  sortInput?.addEventListener("change",()=>{assetOffset=0;loadAssets();});
  $("#assetPrev")?.addEventListener("click",()=>{assetOffset=Math.max(0,assetOffset-assetPageSize);loadAssets();});
  $("#assetNext")?.addEventListener("click",()=>{if(assetOffset+assetPageSize<assetTotal){assetOffset+=assetPageSize;loadAssets();}});
  async function deleteAsset(asset,button) {
    if(!confirm("Permanently delete “"+asset.title+"” from the database and local output folder?"))return;
    button.disabled=true;if(libraryStatus)libraryStatus.textContent="Deleting saved media…";
    try{
      await readJson(await fetch("/api/assets/"+asset.id,{method:"DELETE"}));
      assetCache.delete(asset.id);
      assetTotal=Math.max(0,assetTotal-1);
      if(assetOffset>=assetTotal&&assetOffset>0)assetOffset=Math.max(0,assetOffset-assetPageSize);
      await loadAssets();
      if(libraryStatus)libraryStatus.textContent="Deleted “"+asset.title+"”.";
    }catch(error){if(libraryStatus)libraryStatus.textContent=error.message;}
    finally{button.disabled=false;}
  }
  const editDialog=$("#editAssetDialog");
  function openEditAsset(asset) {
    $("#editAssetId").value=asset.id;$("#editAssetTitle").value=asset.title;
    $("#editAssetPrompt").value=asset.prompt||"";$("#editAssetNegative").value=asset.negative_prompt||"";
    if($("#editAssetStatus"))$("#editAssetStatus").textContent="";
    editDialog?.showModal();
  }
  function closeEditAsset(){if(editDialog?.open)editDialog.close();}
  $("#closeEditAsset")?.addEventListener("click",closeEditAsset);
  $("#cancelEditAsset")?.addEventListener("click",closeEditAsset);
  $("#editAssetForm")?.addEventListener("submit",async event=>{
    event.preventDefault();const id=$("#editAssetId").value;const button=$("#saveAssetDetails");const status=$("#editAssetStatus");
    button.disabled=true;if(status)status.textContent="Saving asset details…";
    try{
      const asset=await readJson(await fetch("/api/assets/"+id,{method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:$("#editAssetTitle").value,prompt:$("#editAssetPrompt").value,negative_prompt:$("#editAssetNegative").value})}));
      assetCache.set(asset.asset.id,asset.asset);await loadAssets();
      if(libraryStatus)libraryStatus.textContent="Saved changes to “"+asset.asset.title+"”.";
      closeEditAsset();
    }catch(error){if(status)status.textContent=error.message;}
    finally{button.disabled=false;}
  });

  async function mutateDatasetPhoto(photo, action, payload) {
    const response=await fetch("/api/face-photos/"+encodeURIComponent(photo.id),{
      method:action,headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)
    });
    return readJson(response);
  }
  let activeDatasetType="all";
  const datasetTypeLabels={all:"All groups",face_angles:"Face angles",expressions:"Expressions",body_type:"Body type · global"};
  $$(".dataset-category-tab").forEach(tab=>tab.addEventListener("click",()=>{
    activeDatasetType=tab.dataset.datasetFilter||"face_angles";
    $$(".dataset-category-tab").forEach(item=>item.classList.toggle("active",item===tab));
    loadDatasetGroups();
  }));
  let editingTrainingPhoto=null;
  let latestTrainingPhotoPreview=null;
  const trainingPhotoDialog=$("#editTrainingPhotoDialog");
  function openTrainingPhotoEdit(photo,group) {
    editingTrainingPhoto={photo,group};latestTrainingPhotoPreview=null;
    $("#trainingPhotoOldPreview").src=photo.content_url;
    const next=$("#trainingPhotoNewPreview");next.removeAttribute("src");next.hidden=true;
    $("#trainingPhotoNewPlaceholder").hidden=false;
    $("#trainingPhotoEditPrompt").value="";
    $("#trainingPhotoEditNegative").value="different person, changed hairstyle, changed apparent age, cartoon, CGI, blur, watermark";
    $("#trainingPhotoEditStrength").value=group.dataset_type==="expressions"?"0.30":"0.35";
    $("#trainingPhotoEditSteps").value="20";$("#saveNewTrainingPhoto").disabled=true;
    $("#trainingPhotoEditStatus").textContent="";$("#trainingPhotoUseBodyModel").checked=false;
    if(trainingPhotoDialog&&!trainingPhotoDialog.open)trainingPhotoDialog.showModal();
  }
  function closeTrainingPhotoEdit(){if(trainingPhotoDialog?.open)trainingPhotoDialog.close();}
  $("#closeTrainingPhotoEdit")?.addEventListener("click",closeTrainingPhotoEdit);
  $("#keepOriginalTrainingPhoto")?.addEventListener("click",closeTrainingPhotoEdit);
  $("#editTrainingPhotoForm")?.addEventListener("submit",async event=>{
    event.preventDefault();if(!editingTrainingPhoto)return;
    const status=$("#trainingPhotoEditStatus"),button=$("#generateTrainingPhotoPreview");
    button.disabled=true;if(status)status.textContent="Generating preview. The saved photo will not change yet…";
    try{
      const formData=new FormData();
      formData.set("prompt",$("#trainingPhotoEditPrompt").value);
      formData.set("negative_prompt",$("#trainingPhotoEditNegative").value);
      formData.set("strength",$("#trainingPhotoEditStrength").value);
      formData.set("steps",$("#trainingPhotoEditSteps").value);
      formData.set("image_preset","photorealistic");formData.set("aspect_ratio","square");
      formData.set("seed",String(Math.floor(Math.random()*2147483648)));formData.set("task_id",newGenerationTaskId());
      if($("#trainingPhotoUseBodyModel").checked)formData.set("use_global_body_model","true");
      addComputeData(formData);
      const result=await readJson(await fetch("/api/face-photos/"+encodeURIComponent(editingTrainingPhoto.photo.id)+"/edit-preview",{method:"POST",body:formData}));
      latestTrainingPhotoPreview=result.image_data_url;
      const next=$("#trainingPhotoNewPreview");next.src=latestTrainingPhotoPreview;next.hidden=false;
      $("#trainingPhotoNewPlaceholder").hidden=true;$("#saveNewTrainingPhoto").disabled=false;
      if(status)status.textContent="Preview ready. Compare both photos, then choose Keep original or Save new photo.";
    }catch(error){if(status)status.textContent=error.message;}
    finally{button.disabled=false;}
  });
  $("#saveNewTrainingPhoto")?.addEventListener("click",async()=>{
    if(!editingTrainingPhoto||!latestTrainingPhotoPreview)return;
    const status=$("#trainingPhotoEditStatus"),button=$("#saveNewTrainingPhoto");button.disabled=true;
    if(status)status.textContent="Updating only this training photo…";
    try{
      const blob=await (await fetch(latestTrainingPhotoPreview)).blob();
      const formData=new FormData();formData.append("photo",blob,"edited-training-photo.png");
      await readJson(await fetch("/api/face-photos/"+encodeURIComponent(editingTrainingPhoto.photo.id),{method:"PUT",body:formData}));
      if(status)status.textContent="Saved. Only this selected training photo was replaced; every other photo remains unchanged.";
      await loadDatasetGroups();await refreshProfiles();closeTrainingPhotoEdit();
    }catch(error){if(status)status.textContent="Could not save the edited photo: "+error.message;}
    finally{button.disabled=false;}
  });

  async function loadDatasetGroups() {
    const container=$("#datasetGroups");
    if(!container)return;
    const countLabel=$("#datasetGroupCount");
    const status=$("#datasetStatus");
    if(countLabel)countLabel.textContent="Loading training datasets…";
    try{
      const data=await readJson(await fetch("/api/face-dataset-groups",{cache:"no-store"}));
      const allGroups=data.groups||[];
      const groups=activeDatasetType==="all"?allGroups:allGroups.filter(group=>(group.dataset_type||"face_angles")===activeDatasetType);
      $$(".dataset-category-tab").forEach(tab=>{
        const amount=tab.dataset.datasetFilter==="all"?allGroups.length:allGroups.filter(group=>(group.dataset_type||"face_angles")===tab.dataset.datasetFilter).length;
        tab.textContent=(datasetTypeLabels[tab.dataset.datasetFilter]||tab.dataset.datasetFilter)+" ("+amount+")";
      });
      container.replaceChildren();
      if(countLabel)countLabel.textContent=groups.length+" "+(datasetTypeLabels[activeDatasetType]||"")+" dataset group(s) · "+allGroups.length+" total";
      if(!groups.length){
        const empty=document.createElement("div");empty.className="empty-state panel";
        const title=document.createElement("h3");title.textContent="No "+(datasetTypeLabels[activeDatasetType]||"")+" datasets yet";
        const note=document.createElement("p");
        note.textContent=activeDatasetType==="all"
          ?"Create a face-angle, expression, or body-type group from Face models. Saved groups remain listed even when they contain no photos."
          :activeDatasetType==="body_type"
          ?"Open Face models → Body type · global to generate prompt-based samples without uploading a reference photo."
          :activeDatasetType==="expressions"
          ?"Open Face models → Expressions to upload a source face and generate expression variations."
          :"Open Face models → Face angles to upload a source face and generate viewpoint variations.";
        empty.append(title,note);container.appendChild(empty);return;
      }
      groups.forEach(group=>{
        const busy=group.status==="generating";
        const card=document.createElement("article");card.className="panel dataset-group-card";
        const header=document.createElement("div");header.className="dataset-group-heading";
        const title=document.createElement("h2");title.textContent=group.title;
        const state=document.createElement("span");state.className="dataset-group-status "+(group.status||"");state.textContent=group.status||"unknown";
        const headingCopy=document.createElement("div");headingCopy.append(title);
        const caption=document.createElement("p");caption.className="dataset-group-caption";
        const date=group.created_at?new Date(group.created_at).toLocaleString():"Saved";
        caption.textContent=(group.profile_name||"Face profile")+" · "+(group.photo_count||0)+" photo(s) · "+(group.generated_count||0)+"/"+(group.expected_count||5)+" generated · "+date;
        headingCopy.append(caption);
        if(group.prompt){
          const promptNote=document.createElement("p");promptNote.className="dataset-group-prompt";
          promptNote.textContent="Prompt: "+group.prompt;headingCopy.append(promptNote);
        }
        header.append(headingCopy,state);card.append(header);
        const photos=document.createElement("div");photos.className="dataset-group-photos";
        (group.photos||[]).forEach(photo=>{
          const figure=document.createElement("div");figure.className="dataset-group-photo";
          const image=document.createElement("img");image.src=photo.content_url;image.alt=photo.angle_label||"Training photo";image.loading="lazy";
          image.addEventListener("error",()=>{image.alt="Preview unavailable";image.style.opacity=".35";});
          const label=document.createElement("span");label.textContent=photo.angle_label||photo.filename||"Training photo";
          const validation=document.createElement("span");
          const validationStatus=photo.validation_status||"not_checked";
          validation.className="photo-validation-badge "+validationStatus.replace(/_/g,"-");
          validation.textContent=validationStatus==="passed"?"Pose verified":validationStatus==="needs_review"?"Needs review":validationStatus==="not_applicable"?"Source photo":"Not checked";
          if(photo.validation_message)validation.title=photo.validation_message;
          const validationDetail=document.createElement("span");validationDetail.className="photo-validation-details";
          if(photo.validation_message)validationDetail.textContent=photo.validation_message;
          else if(Number.isFinite(Number(photo.actual_yaw)))validationDetail.textContent="Estimated yaw "+Number(photo.actual_yaw).toFixed(0)+"°, pitch "+Number(photo.actual_pitch||0).toFixed(0)+"°.";
          const photoActions=document.createElement("div");photoActions.className="dataset-photo-actions";
          const renamePhoto=document.createElement("button");renamePhoto.type="button";renamePhoto.className="secondary-button";renamePhoto.textContent="Edit label";renamePhoto.disabled=busy;
          renamePhoto.addEventListener("click",async()=>{
            const next=window.prompt("Photo label",photo.angle_label||"Training photo");
            if(next===null)return;
            try{
              await mutateDatasetPhoto(photo,"PATCH",{angle_label:next});
              if(status)status.textContent="Photo label updated.";
              await loadDatasetGroups();
            }catch(error){if(status)status.textContent=error.message;}
          });
          const replaceInput=document.createElement("input");replaceInput.type="file";replaceInput.accept="image/png,image/jpeg,image/webp";replaceInput.hidden=true;replaceInput.setAttribute("aria-label","Choose replacement image for "+label.textContent);
          const replacePhoto=document.createElement("button");replacePhoto.type="button";replacePhoto.className="secondary-button";replacePhoto.textContent="Replace";replacePhoto.disabled=busy;
          replacePhoto.addEventListener("click",()=>replaceInput.click());
          replaceInput.addEventListener("change",async()=>{
            const file=replaceInput.files?.[0];if(!file)return;
            if(file.size>8*1024*1024){if(status)status.textContent="Replacement image must be 8 MB or smaller.";replaceInput.value="";return;}
            replacePhoto.disabled=true;if(status)status.textContent="Replacing "+(photo.angle_label||"training photo")+"…";
            try{
              const form=new FormData();form.append("photo",file,file.name);
              await readJson(await fetch("/api/face-photos/"+encodeURIComponent(photo.id),{method:"PUT",body:form}));
              if(status)status.textContent="Photo replaced. The training dataset file and database copy are updated.";
              await loadDatasetGroups();
            }catch(error){if(status)status.textContent=error.message;}
            finally{replacePhoto.disabled=false;}
          });
          const deletePhoto=document.createElement("button");deletePhoto.type="button";deletePhoto.className="secondary-button danger-button";deletePhoto.textContent="Delete photo";deletePhoto.disabled=busy;
          deletePhoto.addEventListener("click",async()=>{
            if(!window.confirm("Delete this photo from the training group and profile dataset? The original in Generated images will remain unchanged."))return;
            deletePhoto.disabled=true;if(status)status.textContent="Deleting training photo…";
            try{
              await readJson(await fetch("/api/face-photos/"+encodeURIComponent(photo.id),{method:"DELETE"}));
              if(status)status.textContent="Training photo deleted. The original generated-library image, if any, was preserved.";
              await loadDatasetGroups();
              await refreshProfiles();
            }catch(error){if(status)status.textContent=error.message;deletePhoto.disabled=false;}
          });
          const editWithPrompt=document.createElement("button");editWithPrompt.type="button";editWithPrompt.className="secondary-button";editWithPrompt.textContent="Edit with prompt";editWithPrompt.disabled=busy;editWithPrompt.addEventListener("click",()=>openTrainingPhotoEdit(photo,group));
          photoActions.append(editWithPrompt,renamePhoto,replacePhoto,deletePhoto);
          figure.append(image,label,validation);
          if(validationDetail.textContent)figure.append(validationDetail);
          figure.append(photoActions,replaceInput);photos.appendChild(figure);
        });
        if(!(group.photos||[]).length){
          const emptyPhotos=document.createElement("div");
          emptyPhotos.className="empty-state dataset-group-empty";
          emptyPhotos.textContent="This group is saved but has no training photos yet. It will remain here until you explicitly delete the group.";
          photos.appendChild(emptyPhotos);
        }
        card.appendChild(photos);
        if(group.error){const error=document.createElement("p");error.className="dataset-group-error";error.textContent=group.error;card.appendChild(error);}
        if(busy){const warning=document.createElement("p");warning.className="field-help";warning.textContent="Generation is still in progress. Group edits are temporarily disabled.";card.appendChild(warning);}
        const actions=document.createElement("div");actions.className="dataset-group-actions";
        const isGlobalBody=group.dataset_type==="body_type";
        const addPhotos=document.createElement("a");addPhotos.className="button-link secondary-link";addPhotos.href="/faces?profile_id="+encodeURIComponent(group.profile_id)+"&group_id="+encodeURIComponent(group.id);addPhotos.textContent="Add photos";addPhotos.hidden=isGlobalBody;
        const renameGroup=document.createElement("button");renameGroup.type="button";renameGroup.className="secondary-button";renameGroup.textContent="Rename group";renameGroup.disabled=busy;
        renameGroup.addEventListener("click",async()=>{
          const next=window.prompt("Dataset group name",group.title);
          if(next===null)return;
          try{
            await readJson(await fetch("/api/face-dataset-groups/"+encodeURIComponent(group.id),{
              method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:next})
            }));
            if(status)status.textContent="Dataset group renamed.";
            await loadDatasetGroups();
          }catch(error){if(status)status.textContent=error.message;}
        });
        const faceLink=document.createElement("a");faceLink.className="button-link secondary-link";faceLink.href="/faces?profile_id="+encodeURIComponent(group.profile_id);faceLink.textContent="Train this profile";faceLink.hidden=isGlobalBody;
        const imageLink=document.createElement("a");imageLink.className="button-link";imageLink.href="/generate/image?profile_id="+encodeURIComponent(group.profile_id);imageLink.textContent="Generate with face";imageLink.hidden=isGlobalBody;
        const deleteGroup=document.createElement("button");deleteGroup.type="button";deleteGroup.className="secondary-button danger-button";deleteGroup.textContent="Delete entire group";deleteGroup.disabled=busy;
        deleteGroup.addEventListener("click",async()=>{
          const confirmText="Delete the dataset group “"+group.title+"” and all "+(group.photo_count||0)+" grouped training photo(s)? This will not delete the original images from Generated images or the face profile itself.";
          if(!window.confirm(confirmText))return;
          deleteGroup.disabled=true;if(status)status.textContent="Deleting dataset group and its training photo copies…";
          try{
            const result=await readJson(await fetch("/api/face-dataset-groups/"+encodeURIComponent(group.id),{method:"DELETE"}));
            if(status)status.textContent="Deleted the dataset group and "+result.deleted_photos+" photo(s). Original generated-library images were kept.";
            await loadDatasetGroups();
            await refreshProfiles();
          }catch(error){if(status)status.textContent=error.message;deleteGroup.disabled=false;}
        });
        actions.append(addPhotos,renameGroup,faceLink,imageLink,deleteGroup);card.appendChild(actions);container.appendChild(card);
      });
      if(status)status.textContent="";
    }catch(error){
      container.replaceChildren();const empty=document.createElement("div");empty.className="empty-state panel";empty.textContent="Could not load training datasets: "+error.message;container.appendChild(empty);
      if(countLabel)countLabel.textContent="Dataset library unavailable";
      if(status)status.textContent=error.message;
    }
  }
  $("#datasetRefresh")?.addEventListener("click",loadDatasetGroups);

  async function loadStatus() {
    try{
      const data=await readJson(await fetch("/health"));
      const gpu=data.local_gpu||{};
      if(gpuStatus && !useGpuAcceleration?.checked)gpuStatus.textContent="GPU acceleration is off. "+(gpu.available?"System GPU detected: "+gpu.name+".":"No local CUDA GPU is available to this PyTorch installation. ")+(data.gpu_worker_configured?"GHive / Colab GPU worker is configured; enable GPU acceleration to check availability.":"GHive / Colab GPU worker is not configured; local CPU remains available.");
    }catch(error){if(gpuStatus)gpuStatus.textContent=error.message;}
  }
  loadStatus();
  if($("#globalBodyModelStatus")||$("#globalBodyGenerationStatus")||$("#useGlobalBodyModel")||$("#datasetGroups"))loadGlobalBodyStatus().catch(()=>{});
  if($("#profileList")||$("#trainProfile")||$("#faceProfile")||$("#videoProfile"))refreshProfiles().catch(error=>{
    const profileList=$("#profileList");if(profileList)profileList.textContent="Could not load profiles: "+error.message;
  });
  loadAssets();
  loadDatasetGroups();
})();
