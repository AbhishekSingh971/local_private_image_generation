(() => {
  "use strict";
  const $ = (selector, root=document) => root.querySelector(selector);
  const $$ = (selector, root=document) => [...root.querySelectorAll(selector)];
  const page = document.body.dataset.page || "dashboard";
  const computeBackend = $("#computeBackend");
  const trainingMethodSelect = $("#trainingMethod");
  const globalSettingsStatus = $("#globalSettingsStatus");
  const useGpuAcceleration = { get checked() { return Boolean(computeBackend && computeBackend.value !== "cpu"); } };
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
    const selected = computeBackend?.value || "auto";
    formData.set("use_gpu", selected === "cpu" ? "false" : "true");
    formData.set("compute_backend", selected);
    formData.delete("use_runpod");
    return formData;
  }
  function applyComputeStatus() {
    if (!gpuStatus || !gpuModePill) return;
    const requested = computeBackend?.value || "auto";
    const selected = requested === "auto" ? (computeAvailability?.auto_backend || "auto") : requested;
    const labels = {cpu:"LOCAL · CPU",local_gpu:"SYSTEM GPU",cloud_gpu:"GHIVE GPU",runpod:"GHIVE GPU",auto:"AUTO · CHECKING"};
    gpuModePill.textContent = requested === "auto" ? (labels[selected] || "AUTO · CHECKING") : (labels[selected] || "DEVICE SELECTED");
    if (selected === "local_gpu") gpuStatus.textContent = "System GPU selected. " + (computeAvailability?.local_gpu?.name || "CUDA GPU") + " will be used for supported tasks.";
    else if (selected === "cloud_gpu" || selected === "runpod") gpuStatus.textContent = "GHive remote GPU selected. Jobs will be sent to the configured authenticated worker.";
    else if (requested === "auto") gpuStatus.textContent = selected === "cpu" ? "Auto found no usable GPU and will use the local CPU. CPU inference may be slow." : "Auto prefers System GPU, then GHive GPU, then local CPU.";
    else gpuStatus.textContent = "Local CPU selected. No GPU worker will be used.";
    if (computeAvailability?.cloud_error && selected !== "cloud_gpu" && selected !== "runpod") gpuStatus.textContent += " GHive unavailable: " + computeAvailability.cloud_error;
  }
  function renderComputeOptions(options) {
    computeAvailability = options;
    if (computeBackend) {
      const available = new Set(options.available_backends || ["cpu"]);
      const localName = options.local_gpu?.name || "System CUDA GPU";
      const cloudName = options.cloud_gpu?.name || "GHive CUDA GPU";
      [...computeBackend.options].forEach(option => {
        option.disabled = !["auto","cpu"].includes(option.value) && !available.has(option.value);
        if (option.value === "local_gpu") {
          option.textContent = available.has("local_gpu") ? localName + " · System GPU" : "System GPU · unavailable";
        } else if (option.value === "cloud_gpu" || option.value === "runpod") {
          option.textContent = available.has("cloud_gpu") ? cloudName + " · GHive" : "GHive GPU · unavailable";
        } else if (option.value === "cpu") {
          option.textContent = "Local CPU";
        } else if (option.value === "auto") {
          option.textContent = "Auto · best available device";
        }
      });
    }
    applyComputeStatus();
  }
  async function saveGlobalSettings(fields) {
    const data = await readJson(await fetch("/api/settings",{method:"PUT",headers:{"Content-Type":"application/json","Accept":"application/json"},body:JSON.stringify(fields)}));
    if (globalSettingsStatus) globalSettingsStatus.textContent = "Global settings saved.";
    return data.settings;
  }
  async function loadGlobalSettings() {
    try {
      const data = await readJson(await fetch("/api/settings",{cache:"no-store"}));
      if (computeBackend && data.settings?.compute_backend) computeBackend.value = data.settings.compute_backend;
      if (trainingMethodSelect && data.settings?.training_method) trainingMethodSelect.value = data.settings.training_method;
      if (globalSettingsStatus) globalSettingsStatus.textContent = "Saved globally · applies across all pages and training jobs.";
    } catch(error) { if(globalSettingsStatus)globalSettingsStatus.textContent="Could not load saved settings: "+error.message; }
  }
  async function probeCloudGpu() {
    if (gpuModePill) gpuModePill.textContent = "CHECKING DEVICE…";
    if (gpuStatus) gpuStatus.textContent = "Checking local CUDA support and the configured remote GPU worker…";
    if (computeBackend) computeBackend.disabled = true;
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(),25000);
    try {
      const response = await fetch("/api/gpu-status?probe=1",{cache:"no-store",signal:controller.signal,headers:{"Accept":"application/json"}});
      renderComputeOptions(await readJson(response));
    } catch(error) {
      const message = error.name==="AbortError" ? "GPU check timed out. Verify GPU_WORKER_URL and the worker token." : error.message || "Could not check GPU availability.";
      renderComputeOptions({available_backends:["cpu"],auto_backend:"cpu",local_gpu:{available:false},cloud_gpu:{available:false},cloud_error:message});
    } finally { window.clearTimeout(timeoutId); if(computeBackend)computeBackend.disabled=false; }
  }
  computeBackend?.addEventListener("change",async()=>{
    applyComputeStatus();
    try { await saveGlobalSettings({compute_backend:computeBackend.value}); }
    catch(error) { if(globalSettingsStatus)globalSettingsStatus.textContent="Could not save compute setting: "+error.message; }
    await loadGlobalBodyStatus(false);
  });
  trainingMethodSelect?.addEventListener("change",async()=>{
    try {
      await saveGlobalSettings({training_method:trainingMethodSelect.value});
      if(trainingMethodSelect.value==="full_finetune" && $("#trainStatus")) $("#trainStatus").textContent="Full fine-tuning is selected globally. Training is blocked until its dedicated trainer is installed; LoRA will not be substituted.";
    } catch(error) { if(globalSettingsStatus)globalSettingsStatus.textContent="Could not save training method: "+error.message; }
  });
  (async()=>{await loadGlobalSettings();await probeCloudGpu();})();

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
    ["faceAngle","faceExpression","useProfileBodyComponent"].forEach(id=>{
      const control=$("#"+id);
      if(!control)return;
      control.disabled=!profileSelected;
      if(!profileSelected)control.value="";
    });
  }
  $("#faceProfile")?.addEventListener("change",updateFaceConditionControls);
  $("#useGlobalBodyModel")?.addEventListener("change",()=>{
    if($("#useGlobalBodyModel").checked&&$("#useProfileBodyComponent"))$("#useProfileBodyComponent").checked=false;
  });
  $("#useProfileBodyComponent")?.addEventListener("change",()=>{
    if($("#useProfileBodyComponent").checked&&$("#useGlobalBodyModel"))$("#useGlobalBodyModel").checked=false;
  });
  function updateVideoFaceConditionControls() {
    const selected=Boolean($("#videoProfile")?.value);
    ["videoFaceAngle","videoFaceExpression","videoUseProfileBodyComponent"].forEach(id=>{
      const control=$("#"+id);
      if(!control)return;
      control.disabled=!selected;
      if(!selected)control.value="";
    });
  }
  $("#videoProfile")?.addEventListener("change",updateVideoFaceConditionControls);
  updateVideoFaceConditionControls();

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
        if (profile.local_model_ready) {
          const syncButton = document.createElement("button");
          syncButton.type = "button";
          syncButton.className = "secondary-button";
          syncButton.textContent = profile.runpod_model_ready ? "Resync LoRA to GHive" : "Sync LoRA to GHive";
          syncButton.addEventListener("click", async () => {
            syncButton.disabled = true;
            const previousLabel = syncButton.textContent;
            syncButton.textContent = "Uploading LoRA…";
            try {
              const result = await readJson(await fetch("/api/face-profiles/" + encodeURIComponent(profile.id) + "/sync-cloud-lora", {method:"POST", headers:{"Accept":"application/json","X-Requested-With":"XMLHttpRequest"}}));
              alert(result.message || "LoRA synchronized to GHive.");
              await refreshProfiles();
            } catch (error) {
              alert("GHive LoRA sync failed: " + error.message);
              syncButton.disabled = false;
              syncButton.textContent = previousLabel;
            }
          });
          actions.appendChild(syncButton);
        }
        const manageButton=document.createElement("a");
        manageButton.className="button-link secondary-link";
        manageButton.href="/training-data?profile_id="+encodeURIComponent(profile.id);
        manageButton.textContent="View data & manage photos";
        actions.append(manageButton, useButton, deleteButton);
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
      const requestedBackend=computeBackend?.value||"auto";
      const selectedBackend=requestedBackend==="auto"?(computeAvailability?.auto_backend||"auto"):requestedBackend;
      const backendReady=(selectedBackend==="runpod"||selectedBackend==="cloud_gpu")?info.runpod_model_ready:(selectedBackend==="auto"?false:info.local_model_ready);
      const generationBodyCheckbox=$("#useGlobalBodyModel");
      if(generationBodyCheckbox){generationBodyCheckbox.disabled=!backendReady;if(!backendReady)generationBodyCheckbox.checked=false;}
      if(editBodyCheckbox){editBodyCheckbox.disabled=!backendReady;if(editBodyCheckbox.disabled)editBodyCheckbox.checked=false;}
      const trainBodyButton=$("#trainGlobalBodyButton");
      if(trainBodyButton)trainBodyButton.disabled=Number(info.image_count||0)<5 || info.local_training_state==="running" || info.runpod_training_state==="running" || Boolean(info.generation_running);
      const clearBodyButton=$("#clearGlobalBodyButton");
      if(clearBodyButton)clearBodyButton.disabled=info.local_training_state==="running" || info.runpod_training_state==="running" || Boolean(info.generation_running) || Boolean(generateBodyDatasetButton?.disabled);
      return info;
    } catch(error) {
      if(showError&&bodyStatus)bodyStatus.textContent="Could not read shared body-model status: "+error.message;
      if(generateStatus)generateStatus.textContent="Could not read shared body-model status. Check the database connection."; 
      return null;
    }
  }
  $("#trainingPhotoUseBodyModel")?.addEventListener("change",event=>{event.target.dataset.userTouched="1";});
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

  const clearGlobalBodyButton=$("#clearGlobalBodyButton");
  clearGlobalBodyButton?.addEventListener("click",async()=>{
    const status=$("#bodyDatasetStatus")||$("#datasetStatus");
    if(!window.confirm("Clear all shared body training photos, dataset groups, saved adapter artifacts and local checkpoints? The shared profile and globalbody token will remain."))return;
    clearGlobalBodyButton.disabled=true;
    if(status)status.textContent="Clearing shared body training data and saved adapters…";
    try{
      const result=await readJson(await fetch("/api/global-body/training-data",{method:"DELETE"}));
      if(status){
        status.textContent="Shared body data cleared: "+result.photos_deleted+" photo(s), "+result.groups_deleted+" group(s), "+result.adapters_deleted+" saved adapter artifact(s). The shared profile was preserved.";
        if(result.remote_cleanup==="deleted")status.textContent+=" GHive worker photos and adapter were also removed.";
        else if(result.remote_cleanup==="unavailable")status.textContent+=" GHive cleanup is pending: "+(result.remote_cleanup_error||"the worker could not be reached")+". Reconnect/update the worker and run Clear again to remove its cached copy.";
        else if(result.remote_cleanup==="not_configured")status.textContent+=" No GHive worker is configured, so only local/database copies were cleared.";
      }
      await loadDatasetGroups();
      await refreshProfiles();
      await loadGlobalBodyStatus();
    }catch(error){
      if(status)status.textContent="Could not clear shared body data: "+error.message;
    }finally{
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
  async function pollComponentTraining(profileId, jobId, backend, purpose) {
    const started=Date.now();
    while(Date.now()-started<8*60*60*1000){
      await new Promise(resolve=>setTimeout(resolve,4000));
      const url="/api/face-profiles/"+profileId+"/component-train-status/"+jobId+"?backend="+backend;
      const data=await readJson(await fetch(url,{cache:"no-store"}));
      const job=data.job||{};
      if(job.status==="completed"){
        const status=$("#trainStatus");
        if(status)status.textContent=purpose+" component training completed. It is stored separately from the identity LoRA.";
        return;
      }
      if(job.status==="failed")throw new Error(job.error||purpose+" component training failed.");
      const status=$("#trainStatus");
      if(status)status.textContent=purpose+" component training is "+(job.status||"running")+"…";
    }
    throw new Error("Stopped polling after 8 hours. Check the GPU worker logs.");
  }
  $("#syncComponentButton")?.addEventListener("click",async()=>{
    const id=$("#trainProfile")?.value;
    const purpose=$("#trainingPurpose")?.value||"identity";
    const status=$("#trainStatus");
    const button=$("#syncComponentButton");
    if(!id){if(status)status.textContent="Select a face profile first.";return;}
    if(purpose==="identity"){if(status)status.textContent="Use the existing Sync cloud LoRA action for the identity adapter.";return;}
    button.disabled=true;
    try{
      const data=await readJson(await fetch("/api/face-profiles/"+id+"/sync-cloud-component/"+purpose,{method:"POST"}));
      if(status)status.textContent=data.message||"Component adapter synced to GHive.";
    }catch(error){if(status)status.textContent=error.message;}
    finally{button.disabled=false;}
  });
  trainForm?.addEventListener("submit",async event=>{
    event.preventDefault();const id=$("#trainProfile")?.value;
    const status=$("#trainStatus");const button=$("#trainButton");
    const profile=profiles.find(item=>item.id===id);
    const purpose=$("#trainingPurpose")?.value||"identity";
    if(!id||!profile){if(status)status.textContent="Create or select a face profile first.";return;}
    button.disabled=true;if(status)status.textContent="Checking eligible "+purpose+" photos and submitting the training job…";
    const backend=computeBackend?.value||"auto";
    try{
      const data=await readJson(await fetch("/api/face-profiles/"+id+"/train",{method:"POST",body:addComputeData(new FormData(trainForm))}));
      if(status)status.textContent=data.message||"Training started.";
      await refreshProfiles();
      const resolvedBackend=data.backend||backend;
      const selectedBackend=(resolvedBackend==="runpod"||resolvedBackend==="cloud_gpu")?"runpod":"local";
      if(purpose==="identity"||profile.id===globalBodyStatusCache?.profile_id){
        await pollTraining(id,selectedBackend);
      }else{
        await pollComponentTraining(id,data.job_id,selectedBackend,purpose);
      }
    }catch(error){if(status)status.textContent=error.message;}
    finally{button.disabled=false;await refreshProfiles().catch(()=>{});}
  });

  const referenceImage=$("#referenceImage");
  referenceImage?.addEventListener("change",()=>{
    const file=referenceImage.files[0];const preview=$("#sourcePreview");const placeholder=$("#sourcePlaceholder");
    if(!file){preview.hidden=true;preview.removeAttribute("src");placeholder.hidden=false;return;}
    preview.src=URL.createObjectURL(file);preview.hidden=false;placeholder.hidden=true;
  });
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
  const stepsControl = $("#steps");
  generationProfile?.addEventListener("change", () => {
    if (generationProfile.value === "preserve_reference") {
      if (stepsControl) stepsControl.value = "25";
    } else if (generationProfile.value === "photorealistic") {
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
    if(status)status.textContent=useGpuAcceleration?.checked?"Sending this image task to the selected compute device…":"Loading the model and generating locally. CPU inference may be slow.";
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
  const videoDurationSlider=$("#videoDurationSlider");
  const videoDurationNumber=$("#videoDurationNumber");
  const videoDurationLabel=$("#videoDurationLabel");
  const videoFps=$("#videoFps");
  const videoSegmentEstimate=$("#videoSegmentEstimate");
  function updateVideoDurationControls(value){
    const duration=Math.max(1,Math.min(60,Math.round(Number(value)||3)));
    if(videoDurationSlider)videoDurationSlider.value=String(duration);
    if(videoDurationNumber)videoDurationNumber.value=String(duration);
    if(videoDurationLabel)videoDurationLabel.textContent=duration+" second"+(duration===1?"":"s");
    const fps=Number(videoFps?.value||8);
    const frames=duration*fps;
    const segments=Math.max(1,Math.ceil(frames/24));
    if(videoSegmentEstimate)videoSegmentEstimate.textContent=duration+" seconds at "+fps+" FPS · "+frames+" frames · "+segments+" AnimateDiff segment"+(segments===1?"":"s")+".";
  }
  videoDurationSlider?.addEventListener("input",()=>updateVideoDurationControls(videoDurationSlider.value));
  videoDurationNumber?.addEventListener("input",()=>{if(videoDurationNumber.value!=="")updateVideoDurationControls(videoDurationNumber.value);});
  videoDurationNumber?.addEventListener("change",()=>updateVideoDurationControls(videoDurationNumber.value));
  videoFps?.addEventListener("change",()=>updateVideoDurationControls(videoDurationNumber?.value||3));
  updateVideoDurationControls(videoDurationNumber?.value||3);
  const videoForm=$("#videoForm");
  videoForm?.addEventListener("submit",async event=>{
    event.preventDefault();
    const button=$("#videoButton");const status=$("#videoStatus");
    const taskId=newGenerationTaskId();
    const progressController=new AbortController();
    button.disabled=true;button.textContent="Generating video…";
    const progress=$("#videoProgress");if(progress)progress.hidden=false;
    updateGenerationProgress("video",{status:"starting",progress:0,stage:"starting",message:"Submitting video generation request…"});
    if(status)status.textContent=useGpuAcceleration?.checked?"Sending this video task to the selected compute device…":"Checking local CUDA and video model files…";
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
      const feedbackPanel=document.createElement("section");feedbackPanel.className="asset-feedback";
      const feedbackHeading=document.createElement("strong");feedbackHeading.textContent="Rate this result";feedbackPanel.appendChild(feedbackHeading);
      const feedbackFields=document.createElement("div");feedbackFields.className="feedback-fields";
      const feedbackDimensions=[
        ["identity","Face identity"],["angle","Face angle"],["expression","Expression"],
        ["body_proportions","Body proportions"],["image_quality","Image quality"],["prompt_adherence","Prompt adherence"]
      ];
      feedbackDimensions.forEach(([key,label])=>{
        const wrap=document.createElement("label");wrap.className="feedback-field";
        const caption=document.createElement("span");caption.textContent=label;
        const select=document.createElement("select");select.name=key;select.setAttribute("aria-label",label+" rating");
        const empty=document.createElement("option");empty.value="";empty.textContent="Not rated";select.appendChild(empty);
        ["1 · Poor","2 · Below average","3 · Acceptable","4 · Good","5 · Excellent"].forEach((text,index)=>{
          const option=document.createElement("option");option.value=String(index+1);option.textContent=text;select.appendChild(option);
        });
        wrap.append(caption,select);feedbackFields.appendChild(wrap);
      });
      feedbackPanel.appendChild(feedbackFields);
      const feedbackActions=document.createElement("div");feedbackActions.className="asset-feedback-actions";
      const submitFeedback=document.createElement("button");submitFeedback.type="button";submitFeedback.className="secondary-button";submitFeedback.textContent="Submit feedback";
      const feedbackStatus=document.createElement("span");feedbackStatus.className="asset-feedback-status";feedbackStatus.setAttribute("aria-live","polite");
      submitFeedback.addEventListener("click",async()=>{
        const scores={};
        feedbackFields.querySelectorAll("select").forEach(select=>{if(select.value)scores[select.name]=Number(select.value);});
        if(!Object.keys(scores).length){feedbackStatus.textContent="Rate at least one dimension.";return;}
        submitFeedback.disabled=true;feedbackStatus.textContent="Saving feedback…";
        try{
          const response=await fetch("/api/assets/"+asset.id+"/feedback",{
            method:"POST",headers:{"Content-Type":"application/json"},
            body:JSON.stringify({scores})
          });
          const data=await readJson(response);
          feedbackStatus.textContent="Feedback saved · reward "+Number(data.result.evaluation.reward).toFixed(2);
        }catch(error){feedbackStatus.textContent="Could not save: "+error.message;}
        finally{submitFeedback.disabled=false;}
      });
      feedbackActions.append(submitFeedback,feedbackStatus);feedbackPanel.appendChild(feedbackActions);
      card.appendChild(feedbackPanel);
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
  let activeDatasetProfileId=new URLSearchParams(window.location.search).get("profile_id")||"";
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

  function renderTrainingProfileCards(profiles, groups) {
    const container=$("#trainingDatasetProfiles");
    if(!container)return;
    container.replaceChildren();
    const allButton=document.createElement("button");
    allButton.type="button";
    allButton.className="secondary-button"+(!activeDatasetProfileId?" active":"");
    allButton.textContent="All profiles ("+profiles.length+")";
    allButton.addEventListener("click",()=>{activeDatasetProfileId="";loadDatasetGroups();});
    container.appendChild(allButton);
    profiles.forEach(profile=>{
      const profileGroups=groups.filter(group=>group.profile_id===profile.id);
      const card=document.createElement("article");
      card.className="panel training-profile-card";
      const heading=document.createElement("div");
      heading.className="training-profile-heading";
      const name=document.createElement("h3");
      name.textContent=profile.name||"Unnamed face profile";
      const count=document.createElement("span");
      count.className="dataset-group-status";
      count.textContent=Number(profile.image_count||0)+" photo(s)";
      heading.append(name,count);
      const token=document.createElement("p");
      token.className="field-help";
      token.textContent="Trigger token: "+(profile.token||"Not set")+" · "+profileGroups.length+" dataset group(s)";
      const training=document.createElement("p");
      training.className="field-help";
      training.textContent="Local: "+(profile.local_model_ready?"trained":(profile.local_training_state||"not trained"))+" · Cloud: "+(profile.runpod_model_ready?"trained":(profile.runpod_training_state||"not trained"));
      const empty=document.createElement("p");
      empty.className="field-help";
      empty.textContent=Number(profile.image_count||0)===0?"No training photos yet. This profile is saved and will remain visible until you delete it.":(profileGroups.length===0?"No dataset groups yet. Its profile photos are still available for training.":"");
      const select=document.createElement("button");
      select.type="button";
      select.className="secondary-button";
      select.textContent=activeDatasetProfileId===profile.id?"Hide profile data":"Show this profile's data";
      select.addEventListener("click",()=>{
        activeDatasetProfileId=activeDatasetProfileId===profile.id?"":profile.id;
        const nextUrl=new URL(window.location.href);
        if(activeDatasetProfileId)nextUrl.searchParams.set("profile_id",activeDatasetProfileId);
        else nextUrl.searchParams.delete("profile_id");
        history.replaceState(null,"",nextUrl.pathname+(nextUrl.search?nextUrl.search:""));
        loadDatasetGroups();
      });
      const manage=document.createElement("a");
      manage.className="button-link secondary-link";
      manage.href="/faces?profile_id="+encodeURIComponent(profile.id);
      manage.textContent="Open face profile";
      card.append(heading,token,training);
      if(empty.textContent)card.appendChild(empty);
      card.append(select,manage);
      container.appendChild(card);
    });
  }

  function renderSelectedProfilePhotos(profile, photos) {
    const section=$("#selectedProfilePhotoSection");
    const heading=$("#selectedProfilePhotoHeading");
    const container=$("#selectedProfilePhotos");
    const status=$("#datasetStatus");
    if(!section||!container)return;
    if(!profile){
      section.hidden=true;container.replaceChildren();return;
    }
    section.hidden=false;
    const records=photos||[];
    if(heading)heading.textContent=profile.name+" · "+records.length+" photo(s)";
    container.replaceChildren();
    if(!records.length){
      const empty=document.createElement("div");empty.className="empty-state panel";
      empty.textContent="This profile has no saved photos yet. Add photos from the Face models page.";
      container.appendChild(empty);return;
    }
    records.forEach(photo=>{
      const card=document.createElement("article");card.className="panel dataset-group-card profile-photo-record";
      const header=document.createElement("div");header.className="dataset-group-heading";
      const title=document.createElement("h3");title.textContent=photo.angle_label||photo.filename||"Training photo";
      const state=document.createElement("span");state.className="dataset-group-status";
      state.textContent=photo.group_id?"In dataset group":"Profile-level upload";
      header.append(title,state);card.appendChild(header);
      const image=document.createElement("img");image.src=photo.content_url;image.alt=photo.angle_label||"Profile training photo";image.loading="lazy";image.className="profile-photo-preview";image.style.maxWidth="220px";image.style.maxHeight="220px";image.style.objectFit="contain";image.style.borderRadius="12px";
      image.addEventListener("error",()=>{image.alt="Preview unavailable";image.style.opacity=".35";});
      card.appendChild(image);
      const fields=document.createElement("div");fields.className="profile-photo-edit-fields";fields.style.display="grid";fields.style.gap="10px";fields.style.maxWidth="560px";
      const labelWrap=document.createElement("label");labelWrap.textContent="Photo label";
      const labelInput=document.createElement("input");labelInput.type="text";labelInput.maxLength=80;labelInput.value=photo.angle_label||"Training photo";labelWrap.appendChild(labelInput);
      const kindWrap=document.createElement("label");kindWrap.textContent="Training type";
      const kindSelect=document.createElement("select");
      [["manual","Manual / identity"],["source","Source reference"],["angle","Face angle"],["expression","Expression"],["body","Body type"]].forEach(([value,label])=>{
        const option=document.createElement("option");option.value=value;option.textContent=label;kindSelect.appendChild(option);
      });
      kindSelect.value=photo.photo_kind||"manual";kindWrap.appendChild(kindSelect);
      const reviewWrap=document.createElement("label");reviewWrap.textContent="Review status";
      const reviewSelect=document.createElement("select");
      [["not_checked","Not checked"],["passed","Approved"],["needs_review","Needs review"],["not_applicable","Not applicable"]].forEach(([value,label])=>{
        const option=document.createElement("option");option.value=value;option.textContent=label;reviewSelect.appendChild(option);
      });
      reviewSelect.value=photo.validation_status||"not_checked";reviewWrap.appendChild(reviewSelect);
      const captionWrap=document.createElement("label");captionWrap.textContent="Training caption";
      const captionInput=document.createElement("textarea");captionInput.rows=2;captionInput.maxLength=1200;captionInput.value=photo.training_caption||"";captionWrap.appendChild(captionInput);
      fields.append(labelWrap,kindWrap,reviewWrap,captionWrap);card.appendChild(fields);
      if(photo.validation_message){
        const note=document.createElement("p");note.className="field-help";note.textContent=photo.validation_message;card.appendChild(note);
      }
      const actions=document.createElement("div");actions.className="dataset-group-actions";
      const save=document.createElement("button");save.type="button";save.className="secondary-button";save.textContent="Save photo details";
      save.addEventListener("click",async()=>{
        save.disabled=true;
        try{
          await readJson(await fetch(photo.manage_url,{method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify({
            angle_label:labelInput.value,photo_kind:kindSelect.value,validation_status:reviewSelect.value,
            training_caption:captionInput.value,
            validation_message:reviewSelect.value==="passed"?"Approved after manual visual review.":reviewSelect.value==="needs_review"?"Flagged for manual review.":""
          })}));
          if(status)status.textContent="Photo details saved. Training eligibility has been recalculated from the selected type and review status.";
          await loadDatasetGroups();await refreshProfiles();
        }catch(error){if(status)status.textContent="Could not save photo details: "+error.message;}
        finally{save.disabled=false;}
      });
      const replaceInput=document.createElement("input");replaceInput.type="file";replaceInput.accept="image/png,image/jpeg,image/webp";replaceInput.hidden=true;
      const replace=document.createElement("button");replace.type="button";replace.className="secondary-button";replace.textContent="Replace image";
      replace.addEventListener("click",()=>replaceInput.click());
      replaceInput.addEventListener("change",async()=>{
        const file=replaceInput.files?.[0];if(!file)return;
        if(file.size>8*1024*1024){if(status)status.textContent="Replacement image must be 8 MB or smaller.";replaceInput.value="";return;}
        replace.disabled=true;
        try{
          const form=new FormData();form.append("photo",file,file.name);
          await readJson(await fetch(photo.manage_url,{method:"PUT",body:form}));
          if(status)status.textContent="Photo replaced. Its review status was reset to Needs review.";
          await loadDatasetGroups();await refreshProfiles();
        }catch(error){if(status)status.textContent="Could not replace photo: "+error.message;}
        finally{replace.disabled=false;}
      });
      const remove=document.createElement("button");remove.type="button";remove.className="secondary-button danger-button";remove.textContent="Delete photo";
      remove.addEventListener("click",async()=>{
        if(!window.confirm("Delete this training photo from the profile? Its original in Generated images, if present, will remain."))return;
        remove.disabled=true;
        try{
          await readJson(await fetch(photo.manage_url,{method:"DELETE"}));
          if(status)status.textContent="Training photo deleted from this profile.";
          await loadDatasetGroups();await refreshProfiles();
        }catch(error){if(status)status.textContent="Could not delete photo: "+error.message;remove.disabled=false;}
      });
      actions.append(save,replace,remove);card.append(actions,replaceInput);container.appendChild(card);
    });
  }

  async function loadDatasetGroups() {
    const container=$("#datasetGroups");
    if(!container)return;
    const countLabel=$("#datasetGroupCount");
    const status=$("#datasetStatus");
    if(countLabel)countLabel.textContent="Loading training datasets…";
    try{
      const [data,profileData]=await Promise.all([
        readJson(await fetch("/api/face-dataset-groups",{cache:"no-store"})),
        readJson(await fetch("/api/face-profiles",{cache:"no-store",headers:{"Accept":"application/json"}}))
      ]);
      const allGroups=data.groups||[];
      const profiles=profileData.profiles||[];
      const selectedProfile=profiles.find(item=>item.id===activeDatasetProfileId)||null;
      let selectedPhotos=[];
      if(selectedProfile){
        try{
          const photoData=await readJson(await fetch("/api/face-profiles/"+encodeURIComponent(selectedProfile.id)+"/photos",{cache:"no-store"}));
          selectedPhotos=photoData.photos||[];
        }catch(error){
          if(status)status.textContent="Profile loaded, but its photo list could not be read: "+error.message;
        }
      }
      renderSelectedProfilePhotos(selectedProfile,selectedPhotos);
      const profileGroups=activeDatasetProfileId?allGroups.filter(group=>group.profile_id===activeDatasetProfileId):allGroups;
      const groups=activeDatasetType==="all"?profileGroups:profileGroups.filter(group=>(group.dataset_type||"face_angles")===activeDatasetType);
      renderTrainingProfileCards(profiles,allGroups);
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
          const feedbackDetails=document.createElement("details");feedbackDetails.className="dataset-feedback";
          const feedbackSummary=document.createElement("summary");feedbackSummary.textContent="Rate this training sample";feedbackDetails.appendChild(feedbackSummary);
          const feedbackFields=document.createElement("div");feedbackFields.className="feedback-fields dataset-feedback-fields";
          const feedbackDimensions=[["identity","Identity"],["angle","Angle"],["expression","Expression"],["body_proportions","Body proportions"],["image_quality","Image quality"],["prompt_adherence","Prompt adherence"]];
          feedbackDimensions.forEach(([key,labelText])=>{
            const wrap=document.createElement("label");wrap.className="feedback-field";
            const caption=document.createElement("span");caption.textContent=labelText;
            const select=document.createElement("select");select.name=key;select.setAttribute("aria-label",labelText+" rating");
            const empty=document.createElement("option");empty.value="";empty.textContent="Not rated";select.appendChild(empty);
            ["1 · Poor","2 · Below average","3 · Acceptable","4 · Good","5 · Excellent"].forEach((text,index)=>{
              const option=document.createElement("option");option.value=String(index+1);option.textContent=text;select.appendChild(option);
            });
            wrap.append(caption,select);feedbackFields.appendChild(wrap);
          });
          feedbackDetails.appendChild(feedbackFields);
          const submitSampleFeedback=document.createElement("button");submitSampleFeedback.type="button";submitSampleFeedback.className="secondary-button";submitSampleFeedback.textContent="Save sample feedback";submitSampleFeedback.disabled=busy;
          const sampleFeedbackStatus=document.createElement("span");sampleFeedbackStatus.className="asset-feedback-status";sampleFeedbackStatus.setAttribute("aria-live","polite");
          submitSampleFeedback.addEventListener("click",async()=>{
            const scores={};feedbackFields.querySelectorAll("select").forEach(select=>{if(select.value)scores[select.name]=Number(select.value);});
            if(!Object.keys(scores).length){sampleFeedbackStatus.textContent="Rate at least one dimension.";return;}
            submitSampleFeedback.disabled=true;sampleFeedbackStatus.textContent="Saving…";
            try{
              const response=await fetch("/api/face-photos/"+encodeURIComponent(photo.id)+"/feedback",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({scores})});
              const result=await readJson(response);
              sampleFeedbackStatus.textContent="Saved · reward "+Number(result.result.evaluation.reward).toFixed(2);
            }catch(error){sampleFeedbackStatus.textContent="Could not save: "+error.message;}
            finally{submitSampleFeedback.disabled=false;}
          });
          feedbackDetails.append(submitSampleFeedback,sampleFeedbackStatus);
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
          figure.append(feedbackDetails,photoActions,replaceInput);photos.appendChild(figure);
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
      if(gpuStatus && computeBackend?.value==="cpu")gpuStatus.textContent="Local CPU is selected. "+(gpu.available?"System GPU detected: "+gpu.name+".":"No local CUDA GPU is available to this PyTorch installation. ")+(data.gpu_worker_configured?"GHive GPU worker is configured; select it in Global settings to use it.":"GHive GPU worker is not configured; local CPU remains available.");
    }catch(error){if(gpuStatus)gpuStatus.textContent=error.message;}
  }
  loadStatus();
  if($("#globalBodyModelStatus")||$("#globalBodyGenerationStatus")||$("#useGlobalBodyModel")||$("#datasetGroups"))loadGlobalBodyStatus().catch(()=>{});
  if($("#profileList")||$("#trainProfile")||$("#faceProfile")||$("#videoProfile")||$("#addPhotosProfile"))refreshProfiles().catch(error=>{
    const profileList=$("#profileList");if(profileList)profileList.textContent="Could not load profiles: "+error.message;
  });
  loadAssets();
  loadDatasetGroups();
})();
