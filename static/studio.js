(() => {
  "use strict";
  const $ = (selector, root=document) => root.querySelector(selector);
  const $$ = (selector, root=document) => [...root.querySelectorAll(selector)];
  const page = document.body.dataset.page || "dashboard";
  const useRunpod = $("#useRunpod");
  const computeBackendWrap = $("#computeBackendWrap");
  const computeBackend = $("#computeBackend");
  const gpuStatus = $("#gpuStatus");
  const gpuModePill = $("#gpuModePill");
  let computeAvailability = null;
  let profiles = [];
  const assetCache = new Map();
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
    if (useRunpod) {
      formData.set("use_gpu", useRunpod.checked ? "true" : "false");
      formData.set("compute_backend", useRunpod.checked ? (computeBackend?.value || "cpu") : "cpu");
    }
    formData.delete("use_runpod");
    return formData;
  }
  function backendLabel(key, options) {
    if (key === "local_gpu") {
      const gpu = options.local_gpu || {};
      return "System GPU" + (gpu.name ? " · " + gpu.name : "") + (gpu.vram_gb ? " · " + gpu.vram_gb + " GB VRAM" : "");
    }
    if (key === "runpod") {
      const gpu = options.cloud_gpu || {};
      return "RunPod GPU" + (gpu.name ? " · " + gpu.name : "") + (gpu.vram_gb ? " · " + gpu.vram_gb + " GB VRAM" : "");
    }
    return "Local CPU (fallback)";
  }
  function applyComputeStatus() {
    if (!useRunpod || !gpuStatus || !gpuModePill) return;
    const selected = useRunpod.checked ? (computeBackend?.value || "cpu") : "cpu";
    if (!useRunpod.checked) {
      gpuModePill.textContent = "LOCAL · CPU";
      gpuStatus.textContent = "GPU acceleration is off. Requests use the local CPU; no cloud request is made.";
      return;
    }
    const labels = {cpu:"CPU FALLBACK", local_gpu:"SYSTEM GPU", runpod:"CLOUD GPU"};
    gpuModePill.textContent = labels[selected] || "GPU ENABLED";
    if (selected === "local_gpu") {
      gpuStatus.textContent = "Using detected local GPU: " + (computeAvailability?.local_gpu?.name || "CUDA GPU") + ". This avoids cloud compute charges.";
    } else if (selected === "runpod") {
      gpuStatus.textContent = "Using connected RunPod GPU: " + (computeAvailability?.cloud_gpu?.name || "RunPod GPU") + ". This task will be sent to your configured cloud worker.";
    } else {
      gpuStatus.textContent = "No supported GPU was selected; using local CPU. " + (computeAvailability?.cloud_error || "You can choose a GPU if one is available.");
    }
    if (computeAvailability?.cloud_error && selected !== "runpod") gpuStatus.textContent += " RunPod unavailable: " + computeAvailability.cloud_error;
    if (selected === "cpu" && !computeAvailability?.cloud_gpu?.available && !computeAvailability?.local_gpu?.available) {
      gpuStatus.textContent += " No CUDA GPU was detected locally or remotely.";
    }
  }
  function renderComputeOptions(options) {
    computeAvailability = options;
    if (!computeBackend) return;
    const previous = computeBackend.value;
    computeBackend.replaceChildren();
    (options.available_backends || ["cpu"]).forEach(key => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = backendLabel(key, options);
      computeBackend.appendChild(option);
    });
    const fallback = (options.available_backends || ["cpu"]).includes(previous)
      ? previous : (options.auto_backend || "cpu");
    computeBackend.value = fallback;
    applyComputeStatus();
  }
  async function probeRunpod() {
    if (!useRunpod || !computeBackendWrap) return;
    computeBackendWrap.hidden = !useRunpod.checked;
    if (!useRunpod.checked) {
      computeAvailability = null;
      renderComputeOptions({available_backends:["cpu"], auto_backend:"cpu", local_gpu:{}, cloud_gpu:{}, cloud_error:null});
      return;
    }
    if (gpuModePill) gpuModePill.textContent = "CHECKING GPUS…";
    if (gpuStatus) gpuStatus.textContent = "Checking local CUDA support and testing the configured RunPod worker…";
    if (computeBackend) computeBackend.disabled = true;
    try {
      const response = await fetch("/api/gpu-status?probe=1");
      const data = await readJson(response);
      renderComputeOptions(data);
    } catch (error) {
      renderComputeOptions({available_backends:["cpu"], auto_backend:"cpu", local_gpu:{available:false}, cloud_gpu:{available:false}, cloud_error:error.message});
    } finally {
      if (computeBackend) computeBackend.disabled = false;
    }
  }
  if (useRunpod) {
    useRunpod.addEventListener("change", () => {
      computeBackendWrap.hidden = !useRunpod.checked;
      if (useRunpod.checked && computeBackend) computeBackend.value = "auto";
      else if (computeBackend) computeBackend.value = "cpu";
      applyComputeStatus();
      probeRunpod();
    });
    computeBackend?.addEventListener("change", applyComputeStatus);
    computeBackendWrap.hidden = true;
    applyComputeStatus();
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
        + (profile.runpod_model_ready ? " · RunPod trained" : "");
      select.appendChild(option);
    });
    if (profiles.some(profile => profile.id === selected)) select.value = selected;
  }
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
        badges.append(makeTag("RunPod: " + (profile.runpod_model_ready ? "trained" : (profile.runpod_training_state || "not trained")), profile.runpod_model_ready));
        const actions = document.createElement("div"); actions.className = "profile-actions";
        const useButton = document.createElement("button"); useButton.type="button"; useButton.className="secondary-button"; useButton.textContent="Use for training";
        useButton.addEventListener("click", () => {
          if ($("#trainProfile")) $("#trainProfile").value = profile.id;
          if ($("#addPhotosProfile")) $("#addPhotosProfile").value = profile.id;
          if ($("#faceProfile")) $("#faceProfile").value = profile.id;
          if ($("#videoProfile")) $("#videoProfile").value = profile.id;
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
    const response = await fetch("/api/face-profiles");
    const data = await readJson(response);
    profiles = data.profiles || [];
    renderProfiles();
    return profiles;
  }

  const createProfileForm = $("#createProfileForm");
  createProfileForm?.addEventListener("submit", async event => {
    event.preventDefault();
    const button = $("#createProfileButton");
    const status = $("#createProfileStatus");
    if (button) {button.disabled=true;button.textContent="Creating profile…";}
    if (status) status.textContent="Validating photos and saving face profile…";
    try {
      const data = await readJson(await fetch("/api/face-profiles", {method:"POST", body:new FormData(createProfileForm)}));
      createProfileForm.reset();
      await refreshProfiles();
      ["trainProfile","faceProfile","videoProfile","addPhotosProfile"].forEach(id => {const node=$("#"+id);if(node)node.value=data.profile.id;});
      if (status) status.textContent="Created “"+data.profile.name+"” with "+data.profile.image_count+" photo(s). Prompt token: "+data.profile.token+".";
    } catch (error) { if (status) status.textContent=error.message; }
    finally { if (button) {button.disabled=false;button.textContent="Create profile";} }
  });

  const addPhotosButton = $("#addPhotosButton");
  addPhotosButton?.addEventListener("click", async () => {
    const id = $("#addPhotosProfile")?.value;
    const files = $("#morePhotos")?.files;
    const status = $("#addPhotosStatus");
    if (!id) {if(status)status.textContent="Select the profile that should receive these photos.";return;}
    if (!files?.length) {if(status)status.textContent="Choose one or more local photos first.";return;}
    addPhotosButton.disabled=true;
    if(status)status.textContent="Saving selected local photos…";
    const data=new FormData();[...files].forEach(file=>data.append("photos",file));
    try {
      const result=await readJson(await fetch("/api/face-profiles/"+id+"/photos",{method:"POST",body:data}));
      $("#morePhotos").value="";
      await refreshProfiles();
      if(status)status.textContent="Added "+result.added+" photo(s) to "+result.profile.name+". Total: "+result.profile.image_count+".";
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
  $("#generatedPhotoSearch")?.addEventListener("input",()=>{generatedPhotoOffset=0;loadGeneratedPhotoPicker();});
  $("#generatedPhotoPrev")?.addEventListener("click",()=>{generatedPhotoOffset=Math.max(0,generatedPhotoOffset-generatedPhotoLimit);loadGeneratedPhotoPicker();});
  $("#generatedPhotoNext")?.addEventListener("click",()=>{if(generatedPhotoOffset+generatedPhotoLimit<generatedPhotoTotal){generatedPhotoOffset+=generatedPhotoLimit;loadGeneratedPhotoPicker();}});
  $$(".source-tab").forEach(tab=>tab.addEventListener("click",()=>{
    $$(".source-tab").forEach(item=>item.classList.toggle("active",item===tab));
    $$("[data-source-panel]").forEach(panel=>panel.hidden=panel.dataset.sourcePanel!==tab.dataset.sourceTab);
  }));
  $("#addGeneratedPhotosButton")?.addEventListener("click",async()=>{
    const id=$("#addPhotosProfile")?.value;
    const selected=[...selectedGeneratedAssetIds];
    const status=$("#generatedPhotosStatus");const button=$("#addGeneratedPhotosButton");
    if(!id){if(status)status.textContent="Select the face profile that should receive the images.";return;}
    if(!selected.length){if(status)status.textContent="Select at least one generated image.";return;}
    if(selected.length>30){if(status)status.textContent="Select at most 30 generated images per batch.";return;}
    button.disabled=true;if(status)status.textContent="Copying selected database images into the face dataset…";
    try{
      const result=await readJson(await fetch("/api/face-profiles/"+id+"/photos/from-assets",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({asset_ids:selected})}));
      selected.forEach(assetId=>selectedGeneratedAssetIds.delete(assetId));
      await refreshProfiles();
      await loadGeneratedPhotoPicker();
      if(status)status.textContent="Added "+result.added+" generated image(s) to "+result.profile.name+". Total: "+result.profile.image_count+".";
    }catch(error){if(status)status.textContent=error.message;}
    finally{button.disabled=false;updateGeneratedPhotoSelectionCount();}
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
    const backend=useRunpod?.checked?(computeBackend?.value||"cpu"):"cpu";
    try{
      const data=await readJson(await fetch("/api/face-profiles/"+id+"/train",{method:"POST",body:addComputeData(new FormData(trainForm))}));
      if(status)status.textContent=data.message||"Training started.";
      await refreshProfiles();
      await pollTraining(id,backend==="runpod"?"runpod":"local");
    }catch(error){if(status)status.textContent=error.message;}
    finally{button.disabled=false;await refreshProfiles().catch(()=>{});}
  });

  const referenceImage=$("#referenceImage");
  referenceImage?.addEventListener("change",()=>{
    const file=referenceImage.files[0];const preview=$("#sourcePreview");const placeholder=$("#sourcePlaceholder");
    if(!file){preview.hidden=true;preview.removeAttribute("src");placeholder.hidden=false;return;}
    preview.src=URL.createObjectURL(file);preview.hidden=false;placeholder.hidden=true;
  });
  $("#strength")?.addEventListener("input",event=>{const label=$("#strengthValue");if(label)label.textContent=Number(event.target.value).toFixed(2);});

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
      encoding:"Encoding video", saving:"Saving to library", completed:"Complete",
      failed:"Generation failed", busy:"GPU busy"
    };
    if(stageLabel)stageLabel.textContent=stageNames[data.stage]||"Generation in progress";
    if(detail)detail.textContent=data.message||"Working…";
    if(step){
      step.textContent=Number(data.total_steps)>0
        ? "Denoising step "+Number(data.current_step||0)+" of "+Number(data.total_steps)
        : (data.backend==="runpod"?"RunPod GPU":"Local compute");
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

  generationForm?.addEventListener("submit",async event=>{
    event.preventDefault();
    const button=$("#generateButton");const status=$("#generationStatus");
    const taskId=newGenerationTaskId();
    const progressController=new AbortController();
    button.disabled=true;button.textContent="Generating image…";
    const progress=$("#imageProgress");if(progress)progress.hidden=false;
    updateGenerationProgress("image",{status:"starting",progress:0,stage:"starting",message:"Submitting image generation request…"});
    if(status)status.textContent=useRunpod?.checked?"Sending this image task to the selected GPU worker…":"Loading the model and generating locally. CPU inference may be slow.";
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
    if(status)status.textContent=useRunpod?.checked?"Sending this video task to the selected GPU worker…":"Checking local CUDA and video model files…";
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

  async function loadStatus() {
    try{
      const data=await readJson(await fetch("/health"));
      const gpu=data.local_gpu||{};
      if(gpuStatus)gpuStatus.textContent="GPU acceleration is off by default. "+(gpu.available?"System GPU detected: "+gpu.name+".":"No local CUDA GPU is available to this PyTorch installation. ")+(data.runpod_configured?"RunPod is configured; toggle on to check cloud availability.":"RunPod is not configured; local CPU remains available.");
    }catch(error){if(gpuStatus)gpuStatus.textContent=error.message;}
  }
  loadStatus();
  if($("#profileList")||$("#trainProfile")||$("#faceProfile")||$("#videoProfile"))refreshProfiles().catch(error=>{
    const profileList=$("#profileList");if(profileList)profileList.textContent="Could not load profiles: "+error.message;
  });
  loadAssets();
})();
