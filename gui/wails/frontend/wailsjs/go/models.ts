export namespace main {
	
	export class BrowseResult {
	    cancelled: boolean;
	    file: string;
	    folder: string;
	    added: boolean;
	    message: string;
	    error?: supervisor.CLIError;
	
	    static createFrom(source: any = {}) {
	        return new BrowseResult(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.cancelled = source["cancelled"];
	        this.file = source["file"];
	        this.folder = source["folder"];
	        this.added = source["added"];
	        this.message = source["message"];
	        this.error = this.convertValues(source["error"], supervisor.CLIError);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}

}

export namespace supervisor {
	
	export class ActionAvailability {
	    allowed: boolean;
	    reason_code: string;
	    reason: string;
	
	    static createFrom(source: any = {}) {
	        return new ActionAvailability(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.allowed = source["allowed"];
	        this.reason_code = source["reason_code"];
	        this.reason = source["reason"];
	    }
	}
	export class CLIError {
	    code: string;
	    label?: string;
	    component: string;
	    stage: string;
	    retryable: boolean;
	    remedy: string;
	    switch?: string;
	    detail?: string;
	    needed_bytes?: number;
	    available_bytes?: number;
	    sessions?: number;
	    context?: number;
	    per_session_bytes?: number;
	    floor_bytes?: number;
	
	    static createFrom(source: any = {}) {
	        return new CLIError(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.code = source["code"];
	        this.label = source["label"];
	        this.component = source["component"];
	        this.stage = source["stage"];
	        this.retryable = source["retryable"];
	        this.remedy = source["remedy"];
	        this.switch = source["switch"];
	        this.detail = source["detail"];
	        this.needed_bytes = source["needed_bytes"];
	        this.available_bytes = source["available_bytes"];
	        this.sessions = source["sessions"];
	        this.context = source["context"];
	        this.per_session_bytes = source["per_session_bytes"];
	        this.floor_bytes = source["floor_bytes"];
	    }
	}
	export class ActionResult {
	    error?: CLIError;
	
	    static createFrom(source: any = {}) {
	        return new ActionResult(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.error = this.convertValues(source["error"], CLIError);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ActiveModel {
	    catalogue_id?: string;
	    model_path?: string;
	    manifest_path?: string;
	
	    static createFrom(source: any = {}) {
	        return new ActiveModel(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.catalogue_id = source["catalogue_id"];
	        this.model_path = source["model_path"];
	        this.manifest_path = source["manifest_path"];
	    }
	}
	export class Approval {
	    plan_digest: string;
	    components: string[];
	    question?: string;
	
	    static createFrom(source: any = {}) {
	        return new Approval(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.plan_digest = source["plan_digest"];
	        this.components = source["components"];
	        this.question = source["question"];
	    }
	}
	export class BiasEntry {
	    value: number;
	    source: string;
	
	    static createFrom(source: any = {}) {
	        return new BiasEntry(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.value = source["value"];
	        this.source = source["source"];
	    }
	}
	
	export class Profile {
	    name: string;
	    label: string;
	    gui: boolean;
	
	    static createFrom(source: any = {}) {
	        return new Profile(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.name = source["name"];
	        this.label = source["label"];
	        this.gui = source["gui"];
	    }
	}
	export class Capabilities {
	    schema_version: number;
	    prepare_jsonl: boolean;
	    diagnose_json: boolean;
	    status_json: boolean;
	    instance_control: boolean;
	    status_watch: boolean;
	    settings?: boolean;
	    profiles?: Profile[];
	    memory_estimate?: boolean;
	    connection_helpers?: boolean;
	    support_bundle?: boolean;
	    switch_registry?: boolean;
	    request_history?: boolean;
	    per_slot_context?: boolean;
	
	    static createFrom(source: any = {}) {
	        return new Capabilities(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.schema_version = source["schema_version"];
	        this.prepare_jsonl = source["prepare_jsonl"];
	        this.diagnose_json = source["diagnose_json"];
	        this.status_json = source["status_json"];
	        this.instance_control = source["instance_control"];
	        this.status_watch = source["status_watch"];
	        this.settings = source["settings"];
	        this.profiles = this.convertValues(source["profiles"], Profile);
	        this.memory_estimate = source["memory_estimate"];
	        this.connection_helpers = source["connection_helpers"];
	        this.support_bundle = source["support_bundle"];
	        this.switch_registry = source["switch_registry"];
	        this.request_history = source["request_history"];
	        this.per_slot_context = source["per_slot_context"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ClientSnippet {
	    kind: string;
	    title: string;
	    template: string;
	    requires_key: boolean;
	    default_api_key?: string;
	
	    static createFrom(source: any = {}) {
	        return new ClientSnippet(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.kind = source["kind"];
	        this.title = source["title"];
	        this.template = source["template"];
	        this.requires_key = source["requires_key"];
	        this.default_api_key = source["default_api_key"];
	    }
	}
	export class Connection {
	    endpoint: string;
	    model_id: string;
	    context_tokens?: number;
	    authenticated: boolean;
	
	    static createFrom(source: any = {}) {
	        return new Connection(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.endpoint = source["endpoint"];
	        this.model_id = source["model_id"];
	        this.context_tokens = source["context_tokens"];
	        this.authenticated = source["authenticated"];
	    }
	}
	export class Counters {
	    prefill_tokens_total?: number;
	    decode_tokens_total?: number;
	    cached_tokens_total?: number;
	    cache_disk_hits_total?: number;
	    cache_memory_hits_total?: number;
	    cache_misses_total?: number;
	    max_sequence_tokens?: number;
	    draft_rounds_total?: number;
	    draft_tokens_total?: number;
	    draft_accepted_tokens_total?: number;
	    device_lost_total?: number;
	    prefill_seconds_total?: number;
	    decode_seconds_total?: number;
	    completed_successful_requests_total?: number;
	
	    static createFrom(source: any = {}) {
	        return new Counters(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.prefill_tokens_total = source["prefill_tokens_total"];
	        this.decode_tokens_total = source["decode_tokens_total"];
	        this.cached_tokens_total = source["cached_tokens_total"];
	        this.cache_disk_hits_total = source["cache_disk_hits_total"];
	        this.cache_memory_hits_total = source["cache_memory_hits_total"];
	        this.cache_misses_total = source["cache_misses_total"];
	        this.max_sequence_tokens = source["max_sequence_tokens"];
	        this.draft_rounds_total = source["draft_rounds_total"];
	        this.draft_tokens_total = source["draft_tokens_total"];
	        this.draft_accepted_tokens_total = source["draft_accepted_tokens_total"];
	        this.device_lost_total = source["device_lost_total"];
	        this.prefill_seconds_total = source["prefill_seconds_total"];
	        this.decode_seconds_total = source["decode_seconds_total"];
	        this.completed_successful_requests_total = source["completed_successful_requests_total"];
	    }
	}
	export class CurrentModel {
	    catalogue_id: string;
	    display_name: string;
	    model_path: string;
	    manifest_path: string;
	
	    static createFrom(source: any = {}) {
	        return new CurrentModel(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.catalogue_id = source["catalogue_id"];
	        this.display_name = source["display_name"];
	        this.model_path = source["model_path"];
	        this.manifest_path = source["manifest_path"];
	    }
	}
	export class DownloadFolderView {
	    path: string;
	    is_default: boolean;
	    drive: string;
	    free_bytes?: number;
	    change: ActionAvailability;
	
	    static createFrom(source: any = {}) {
	        return new DownloadFolderView(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.path = source["path"];
	        this.is_default = source["is_default"];
	        this.drive = source["drive"];
	        this.free_bytes = source["free_bytes"];
	        this.change = this.convertValues(source["change"], ActionAvailability);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class EffortBiases {
	    low: BiasEntry;
	    medium: BiasEntry;
	    high: BiasEntry;
	    xhigh: BiasEntry;
	
	    static createFrom(source: any = {}) {
	        return new EffortBiases(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.low = this.convertValues(source["low"], BiasEntry);
	        this.medium = this.convertValues(source["medium"], BiasEntry);
	        this.high = this.convertValues(source["high"], BiasEntry);
	        this.xhigh = this.convertValues(source["xhigh"], BiasEntry);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class Endpoints {
	    openai_base_url: string;
	    anthropic_base_url: string;
	
	    static createFrom(source: any = {}) {
	        return new Endpoints(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.openai_base_url = source["openai_base_url"];
	        this.anthropic_base_url = source["anthropic_base_url"];
	    }
	}
	export class EntropyGate {
	    available: boolean;
	
	    static createFrom(source: any = {}) {
	        return new EntropyGate(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.available = source["available"];
	    }
	}
	export class FilePosition {
	    index: number;
	    total: number;
	
	    static createFrom(source: any = {}) {
	        return new FilePosition(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.index = source["index"];
	        this.total = source["total"];
	    }
	}
	export class FitVerdict {
	    class: string;
	    sentence: string;
	    context_tokens: number;
	    sessions: number;
	    free_after_gib?: number;
	    host_free_after_spill_gib?: number;
	    suggested_context?: number;
	
	    static createFrom(source: any = {}) {
	        return new FitVerdict(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.class = source["class"];
	        this.sentence = source["sentence"];
	        this.context_tokens = source["context_tokens"];
	        this.sessions = source["sessions"];
	        this.free_after_gib = source["free_after_gib"];
	        this.host_free_after_spill_gib = source["host_free_after_spill_gib"];
	        this.suggested_context = source["suggested_context"];
	    }
	}
	export class Identity {
	    instance_id: string;
	    pid: number;
	    process_creation_time: string;
	    exe_path: string;
	
	    static createFrom(source: any = {}) {
	        return new Identity(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.instance_id = source["instance_id"];
	        this.pid = source["pid"];
	        this.process_creation_time = source["process_creation_time"];
	        this.exe_path = source["exe_path"];
	    }
	}
	export class StrDefault {
	    value?: string;
	    source: string;
	
	    static createFrom(source: any = {}) {
	        return new StrDefault(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.value = source["value"];
	        this.source = source["source"];
	    }
	}
	export class NumDefault {
	    value?: number;
	    source: string;
	
	    static createFrom(source: any = {}) {
	        return new NumDefault(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.value = source["value"];
	        this.source = source["source"];
	    }
	}
	export class InferenceDefaults {
	    temperature: NumDefault;
	    top_p: NumDefault;
	    top_k: NumDefault;
	    min_p: NumDefault;
	    repeat_penalty: NumDefault;
	    presence_penalty: NumDefault;
	    seed: NumDefault;
	    reasoning_effort: StrDefault;
	    think: StrDefault;
	    max_tokens: NumDefault;
	
	    static createFrom(source: any = {}) {
	        return new InferenceDefaults(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.temperature = this.convertValues(source["temperature"], NumDefault);
	        this.top_p = this.convertValues(source["top_p"], NumDefault);
	        this.top_k = this.convertValues(source["top_k"], NumDefault);
	        this.min_p = this.convertValues(source["min_p"], NumDefault);
	        this.repeat_penalty = this.convertValues(source["repeat_penalty"], NumDefault);
	        this.presence_penalty = this.convertValues(source["presence_penalty"], NumDefault);
	        this.seed = this.convertValues(source["seed"], NumDefault);
	        this.reasoning_effort = this.convertValues(source["reasoning_effort"], StrDefault);
	        this.think = this.convertValues(source["think"], StrDefault);
	        this.max_tokens = this.convertValues(source["max_tokens"], NumDefault);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class JobProgress {
	    bytes_done: number;
	    bytes_total: number;
	
	    static createFrom(source: any = {}) {
	        return new JobProgress(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.bytes_done = source["bytes_done"];
	        this.bytes_total = source["bytes_total"];
	    }
	}
	export class NetworkAddress {
	    adapter: string;
	    ipv4: string;
	    url: string;
	
	    static createFrom(source: any = {}) {
	        return new NetworkAddress(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.adapter = source["adapter"];
	        this.ipv4 = source["ipv4"];
	        this.url = source["url"];
	    }
	}
	export class KeyView {
	    lan_access: boolean;
	    lan_active: boolean;
	    present: boolean;
	    mask: string;
	    restart_needed: boolean;
	    unavailable?: string;
	    copy_with_key: boolean;
	    addresses: NetworkAddress[];
	    addresses_unavailable?: string;
	
	    static createFrom(source: any = {}) {
	        return new KeyView(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.lan_access = source["lan_access"];
	        this.lan_active = source["lan_active"];
	        this.present = source["present"];
	        this.mask = source["mask"];
	        this.restart_needed = source["restart_needed"];
	        this.unavailable = source["unavailable"];
	        this.copy_with_key = source["copy_with_key"];
	        this.addresses = this.convertValues(source["addresses"], NetworkAddress);
	        this.addresses_unavailable = source["addresses_unavailable"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class LastCompletedRequest {
	    prefill_tokens_per_second?: number;
	    decode_tokens_per_second?: number;
	    age_ms?: number;
	    rate_unit: string;
	    age_unit: string;
	
	    static createFrom(source: any = {}) {
	        return new LastCompletedRequest(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.prefill_tokens_per_second = source["prefill_tokens_per_second"];
	        this.decode_tokens_per_second = source["decode_tokens_per_second"];
	        this.age_ms = source["age_ms"];
	        this.rate_unit = source["rate_unit"];
	        this.age_unit = source["age_unit"];
	    }
	}
	export class LifecycleActions {
	    start: ActionAvailability;
	    stop: ActionAvailability;
	    restart: ActionAvailability;
	    prepare: ActionAvailability;
	    download_default: ActionAvailability;
	    check: ActionAvailability;
	    install: ActionAvailability;
	    recover: ActionAvailability;
	    diagnose: ActionAvailability;
	    support_bundle: ActionAvailability;
	
	    static createFrom(source: any = {}) {
	        return new LifecycleActions(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.start = this.convertValues(source["start"], ActionAvailability);
	        this.stop = this.convertValues(source["stop"], ActionAvailability);
	        this.restart = this.convertValues(source["restart"], ActionAvailability);
	        this.prepare = this.convertValues(source["prepare"], ActionAvailability);
	        this.download_default = this.convertValues(source["download_default"], ActionAvailability);
	        this.check = this.convertValues(source["check"], ActionAvailability);
	        this.install = this.convertValues(source["install"], ActionAvailability);
	        this.recover = this.convertValues(source["recover"], ActionAvailability);
	        this.diagnose = this.convertValues(source["diagnose"], ActionAvailability);
	        this.support_bundle = this.convertValues(source["support_bundle"], ActionAvailability);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class LiveSample {
	    instance_id: string;
	    slot_id: number;
	    request_id: number;
	    sample_time: string;
	    phase: string;
	    prefill_tps?: number;
	    decode_tps?: number;
	    elapsed_ms: number;
	    retain_until?: string;
	
	    static createFrom(source: any = {}) {
	        return new LiveSample(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.instance_id = source["instance_id"];
	        this.slot_id = source["slot_id"];
	        this.request_id = source["request_id"];
	        this.sample_time = source["sample_time"];
	        this.phase = source["phase"];
	        this.prefill_tps = source["prefill_tps"];
	        this.decode_tps = source["decode_tps"];
	        this.elapsed_ms = source["elapsed_ms"];
	        this.retain_until = source["retain_until"];
	    }
	}
	export class MemoryAmount {
	    bytes: number;
	    gib: number;
	    evidence: string;
	
	    static createFrom(source: any = {}) {
	        return new MemoryAmount(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.bytes = source["bytes"];
	        this.gib = source["gib"];
	        this.evidence = source["evidence"];
	    }
	}
	export class MemoryGPUBudget {
	    total?: MemoryAmount;
	    dedicated_video_memory?: MemoryAmount;
	    dedicated_video_memory_reason?: string;
	    dedicated_video_memory_source?: string;
	    available?: MemoryAmount;
	    need: MemoryAmount;
	    need_max?: MemoryAmount;
	    reserve: MemoryAmount;
	
	    static createFrom(source: any = {}) {
	        return new MemoryGPUBudget(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.total = this.convertValues(source["total"], MemoryAmount);
	        this.dedicated_video_memory = this.convertValues(source["dedicated_video_memory"], MemoryAmount);
	        this.dedicated_video_memory_reason = source["dedicated_video_memory_reason"];
	        this.dedicated_video_memory_source = source["dedicated_video_memory_source"];
	        this.available = this.convertValues(source["available"], MemoryAmount);
	        this.need = this.convertValues(source["need"], MemoryAmount);
	        this.need_max = this.convertValues(source["need_max"], MemoryAmount);
	        this.reserve = this.convertValues(source["reserve"], MemoryAmount);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class MemoryHostBudget {
	    total?: MemoryAmount;
	    available?: MemoryAmount;
	    need: MemoryAmount;
	    floor: MemoryAmount;
	
	    static createFrom(source: any = {}) {
	        return new MemoryHostBudget(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.total = this.convertValues(source["total"], MemoryAmount);
	        this.available = this.convertValues(source["available"], MemoryAmount);
	        this.need = this.convertValues(source["need"], MemoryAmount);
	        this.floor = this.convertValues(source["floor"], MemoryAmount);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class MemoryRange {
	    min: MemoryAmount;
	    max: MemoryAmount;
	
	    static createFrom(source: any = {}) {
	        return new MemoryRange(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.min = this.convertValues(source["min"], MemoryAmount);
	        this.max = this.convertValues(source["max"], MemoryAmount);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class MemoryShortfall {
	    needed_bytes: number;
	    available_bytes: number;
	    sessions: number;
	    context: number;
	
	    static createFrom(source: any = {}) {
	        return new MemoryShortfall(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.needed_bytes = source["needed_bytes"];
	        this.available_bytes = source["available_bytes"];
	        this.sessions = source["sessions"];
	        this.context = source["context"];
	    }
	}
	export class MemoryUse {
	    rss_bytes?: number;
	    host_available_bytes?: number;
	    unit: string;
	    evidence?: string;
	
	    static createFrom(source: any = {}) {
	        return new MemoryUse(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.rss_bytes = source["rss_bytes"];
	        this.host_available_bytes = source["host_available_bytes"];
	        this.unit = source["unit"];
	        this.evidence = source["evidence"];
	    }
	}
	export class RecommendedContextRow {
	    recommendation_failures?: string[];
	    cache_at_clamp_bytes?: number;
	    host_spill_bytes?: number;
	    free_after_growth_gib?: number;
	    host_free_after_spill_gib?: number;
	    sessions: number;
	    context_per_session: number;
	    fits: boolean;
	    recommended?: boolean;
	    total_min_bytes: number;
	    free_after_gib: number;
	    reason?: string;
	
	    static createFrom(source: any = {}) {
	        return new RecommendedContextRow(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.recommendation_failures = source["recommendation_failures"];
	        this.cache_at_clamp_bytes = source["cache_at_clamp_bytes"];
	        this.host_spill_bytes = source["host_spill_bytes"];
	        this.free_after_growth_gib = source["free_after_growth_gib"];
	        this.host_free_after_spill_gib = source["host_free_after_spill_gib"];
	        this.sessions = source["sessions"];
	        this.context_per_session = source["context_per_session"];
	        this.fits = source["fits"];
	        this.recommended = source["recommended"];
	        this.total_min_bytes = source["total_min_bytes"];
	        this.free_after_gib = source["free_after_gib"];
	        this.reason = source["reason"];
	    }
	}
	export class RecommendedContext {
	    available_bytes: number;
	    available_gib?: number;
	    available_evidence?: string;
	    margin_bytes: number;
	    margin_gib?: number;
	    margin_rule?: string;
	    range_basis: string;
	    step_tokens: number;
	    minimum_context_tokens: number;
	    context_max_tokens: number;
	    max_sessions: number;
	    by_sessions: RecommendedContextRow[];
	
	    static createFrom(source: any = {}) {
	        return new RecommendedContext(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.available_bytes = source["available_bytes"];
	        this.available_gib = source["available_gib"];
	        this.available_evidence = source["available_evidence"];
	        this.margin_bytes = source["margin_bytes"];
	        this.margin_gib = source["margin_gib"];
	        this.margin_rule = source["margin_rule"];
	        this.range_basis = source["range_basis"];
	        this.step_tokens = source["step_tokens"];
	        this.minimum_context_tokens = source["minimum_context_tokens"];
	        this.context_max_tokens = source["context_max_tokens"];
	        this.max_sessions = source["max_sessions"];
	        this.by_sessions = this.convertValues(source["by_sessions"], RecommendedContextRow);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ViewMemoryEstimate {
	    recommendation_failures?: string[];
	    cache_at_clamp_bytes?: number;
	    host_spill_bytes?: number;
	    free_after_growth_gib?: number;
	    host_free_after_spill_gib?: number;
	    table_version: number;
	    evidence: string;
	    byte_unit: string;
	    display_unit: string;
	    fit_verified: boolean;
	    reason: string;
	    weights: MemoryAmount;
	    per_session_state: MemoryAmount;
	    session_state_total: MemoryAmount;
	    sessions: number;
	    context_tokens: number;
	    scratch_phase_allowance: MemoryRange;
	    host_cache_snapshot_allowance: MemoryAmount;
	    reserve: MemoryAmount;
	    total_range: MemoryRange;
	    carve_budget?: MemoryAmount;
	    host_budget?: MemoryAmount;
	    reference_coefficient: Value;
	    formula: string;
	    gpu?: MemoryGPUBudget;
	    host?: MemoryHostBudget;
	    fits?: boolean;
	    fit_reason?: string;
	    fit_scope?: string;
	    cache_mode?: string;
	    device_cache_snapshot_allowance?: MemoryAmount;
	    recommended_context?: RecommendedContext;
	    recommended_context_reason?: string;
	    per_session_vision?: MemoryAmount;
	    vision_session_total?: MemoryAmount;
	    recommended?: boolean;
	    recommendation_reason?: string;
	
	    static createFrom(source: any = {}) {
	        return new ViewMemoryEstimate(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.recommendation_failures = source["recommendation_failures"];
	        this.cache_at_clamp_bytes = source["cache_at_clamp_bytes"];
	        this.host_spill_bytes = source["host_spill_bytes"];
	        this.free_after_growth_gib = source["free_after_growth_gib"];
	        this.host_free_after_spill_gib = source["host_free_after_spill_gib"];
	        this.table_version = source["table_version"];
	        this.evidence = source["evidence"];
	        this.byte_unit = source["byte_unit"];
	        this.display_unit = source["display_unit"];
	        this.fit_verified = source["fit_verified"];
	        this.reason = source["reason"];
	        this.weights = this.convertValues(source["weights"], MemoryAmount);
	        this.per_session_state = this.convertValues(source["per_session_state"], MemoryAmount);
	        this.session_state_total = this.convertValues(source["session_state_total"], MemoryAmount);
	        this.sessions = source["sessions"];
	        this.context_tokens = source["context_tokens"];
	        this.scratch_phase_allowance = this.convertValues(source["scratch_phase_allowance"], MemoryRange);
	        this.host_cache_snapshot_allowance = this.convertValues(source["host_cache_snapshot_allowance"], MemoryAmount);
	        this.reserve = this.convertValues(source["reserve"], MemoryAmount);
	        this.total_range = this.convertValues(source["total_range"], MemoryRange);
	        this.carve_budget = this.convertValues(source["carve_budget"], MemoryAmount);
	        this.host_budget = this.convertValues(source["host_budget"], MemoryAmount);
	        this.reference_coefficient = this.convertValues(source["reference_coefficient"], Value);
	        this.formula = source["formula"];
	        this.gpu = this.convertValues(source["gpu"], MemoryGPUBudget);
	        this.host = this.convertValues(source["host"], MemoryHostBudget);
	        this.fits = source["fits"];
	        this.fit_reason = source["fit_reason"];
	        this.fit_scope = source["fit_scope"];
	        this.cache_mode = source["cache_mode"];
	        this.device_cache_snapshot_allowance = this.convertValues(source["device_cache_snapshot_allowance"], MemoryAmount);
	        this.recommended_context = this.convertValues(source["recommended_context"], RecommendedContext);
	        this.recommended_context_reason = source["recommended_context_reason"];
	        this.per_session_vision = this.convertValues(source["per_session_vision"], MemoryAmount);
	        this.vision_session_total = this.convertValues(source["vision_session_total"], MemoryAmount);
	        this.recommended = source["recommended"];
	        this.recommendation_reason = source["recommendation_reason"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class Value {
	    kind: string;
	    text?: string;
	    boolean?: boolean;
	    items?: Value[];
	    properties?: Record<string, Value>;
	
	    static createFrom(source: any = {}) {
	        return new Value(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.kind = source["kind"];
	        this.text = source["text"];
	        this.boolean = source["boolean"];
	        this.items = this.convertValues(source["items"], Value);
	        this.properties = this.convertValues(source["properties"], Value, true);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ViewModelEstimate {
	    estimate_state: string;
	    reason?: string;
	    evidence_label?: string;
	    fit_verified: boolean;
	    fit_inputs: Record<string, Value>;
	    notes: string[];
	    memory_estimate?: ViewMemoryEstimate;
	    subject: string;
	    subject_kind: string;
	
	    static createFrom(source: any = {}) {
	        return new ViewModelEstimate(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.estimate_state = source["estimate_state"];
	        this.reason = source["reason"];
	        this.evidence_label = source["evidence_label"];
	        this.fit_verified = source["fit_verified"];
	        this.fit_inputs = this.convertValues(source["fit_inputs"], Value, true);
	        this.notes = source["notes"];
	        this.memory_estimate = this.convertValues(source["memory_estimate"], ViewMemoryEstimate);
	        this.subject = source["subject"];
	        this.subject_kind = source["subject_kind"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ModelEstimateResult {
	    estimate?: ViewModelEstimate;
	    fit?: FitVerdict;
	    error?: CLIError;
	
	    static createFrom(source: any = {}) {
	        return new ModelEstimateResult(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.estimate = this.convertValues(source["estimate"], ViewModelEstimate);
	        this.fit = this.convertValues(source["fit"], FitVerdict);
	        this.error = this.convertValues(source["error"], CLIError);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ModelFile {
	    name: string;
	    role: string;
	    bytes: number;
	    sha256: string;
	    url: string;
	    optional: boolean;
	    downloadable: boolean;
	    licence_text?: string;
	
	    static createFrom(source: any = {}) {
	        return new ModelFile(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.name = source["name"];
	        this.role = source["role"];
	        this.bytes = source["bytes"];
	        this.sha256 = source["sha256"];
	        this.url = source["url"];
	        this.optional = source["optional"];
	        this.downloadable = source["downloadable"];
	        this.licence_text = source["licence_text"];
	    }
	}
	export class ModelLicence {
	    id: string;
	    name: string;
	    url: string;
	    prompt_line: string;
	    note?: string;
	    verified_upstream: boolean;
	
	    static createFrom(source: any = {}) {
	        return new ModelLicence(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.name = source["name"];
	        this.url = source["url"];
	        this.prompt_line = source["prompt_line"];
	        this.note = source["note"];
	        this.verified_upstream = source["verified_upstream"];
	    }
	}
	export class Volume {
	    name: string;
	    free_bytes: number;
	    required_bytes: number;
	    bytes_to_write?: number;
	    safety_margin_bytes?: number;
	    free_after_download_bytes?: number;
	
	    static createFrom(source: any = {}) {
	        return new Volume(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.name = source["name"];
	        this.free_bytes = source["free_bytes"];
	        this.required_bytes = source["required_bytes"];
	        this.bytes_to_write = source["bytes_to_write"];
	        this.safety_margin_bytes = source["safety_margin_bytes"];
	        this.free_after_download_bytes = source["free_after_download_bytes"];
	    }
	}
	export class ModelPlanLicence {
	    path: string;
	    text: string;
	    sha256: string;
	
	    static createFrom(source: any = {}) {
	        return new ModelPlanLicence(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.path = source["path"];
	        this.text = source["text"];
	        this.sha256 = source["sha256"];
	    }
	}
	export class ModelPlanFile {
	    file_id: string;
	    url: string;
	    size: number;
	    sha256: string;
	    repository: string;
	    revision: string;
	    path: string;
	    license_id: string;
	    state?: string;
	    replaces: string;
	
	    static createFrom(source: any = {}) {
	        return new ModelPlanFile(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.file_id = source["file_id"];
	        this.url = source["url"];
	        this.size = source["size"];
	        this.sha256 = source["sha256"];
	        this.repository = source["repository"];
	        this.revision = source["revision"];
	        this.path = source["path"];
	        this.license_id = source["license_id"];
	        this.state = source["state"];
	        this.replaces = source["replaces"];
	    }
	}
	export class ModelPlan {
	    id: string;
	    kind: string;
	    plan_digest: string;
	    ready: boolean;
	    files: ModelPlanFile[];
	    licences: ModelPlanLicence[];
	    paths: Record<string, string>;
	    volumes: Volume[];
	    required_bytes: number;
	    free_bytes: number;
	    drive: string;
	    enough_space: boolean;
	    licence?: ModelLicence;
	
	    static createFrom(source: any = {}) {
	        return new ModelPlan(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.kind = source["kind"];
	        this.plan_digest = source["plan_digest"];
	        this.ready = source["ready"];
	        this.files = this.convertValues(source["files"], ModelPlanFile);
	        this.licences = this.convertValues(source["licences"], ModelPlanLicence);
	        this.paths = source["paths"];
	        this.volumes = this.convertValues(source["volumes"], Volume);
	        this.required_bytes = source["required_bytes"];
	        this.free_bytes = source["free_bytes"];
	        this.drive = source["drive"];
	        this.enough_space = source["enough_space"];
	        this.licence = this.convertValues(source["licence"], ModelLicence);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ModelJobError {
	    code: string;
	    remedy: string;
	    exit_code: number;
	    detail?: string;
	
	    static createFrom(source: any = {}) {
	        return new ModelJobError(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.code = source["code"];
	        this.remedy = source["remedy"];
	        this.exit_code = source["exit_code"];
	        this.detail = source["detail"];
	    }
	}
	export class ModelJob {
	    id: string;
	    kind: string;
	    state: string;
	    phase?: string;
	    file_id?: string;
	    file_position?: FilePosition;
	    overall?: JobProgress;
	    bytes_done: number;
	    bytes_total?: number;
	    stalled: boolean;
	    stall_seconds?: number;
	    message?: string;
	    error?: ModelJobError;
	    plan?: ModelPlan;
	    prepared: boolean;
	    manifest_path?: string;
	    repair_output?: string;
	    updated_at: string;
	
	    static createFrom(source: any = {}) {
	        return new ModelJob(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.kind = source["kind"];
	        this.state = source["state"];
	        this.phase = source["phase"];
	        this.file_id = source["file_id"];
	        this.file_position = this.convertValues(source["file_position"], FilePosition);
	        this.overall = this.convertValues(source["overall"], JobProgress);
	        this.bytes_done = source["bytes_done"];
	        this.bytes_total = source["bytes_total"];
	        this.stalled = source["stalled"];
	        this.stall_seconds = source["stall_seconds"];
	        this.message = source["message"];
	        this.error = this.convertValues(source["error"], ModelJobError);
	        this.plan = this.convertValues(source["plan"], ModelPlan);
	        this.prepared = source["prepared"];
	        this.manifest_path = source["manifest_path"];
	        this.repair_output = source["repair_output"];
	        this.updated_at = source["updated_at"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	
	
	
	
	
	export class ModelPlanResult {
	    plan?: ModelPlan;
	    error?: CLIError;
	
	    static createFrom(source: any = {}) {
	        return new ModelPlanResult(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.plan = this.convertValues(source["plan"], ModelPlan);
	        this.error = this.convertValues(source["error"], CLIError);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ModelsDisk {
	    drive: string;
	    path: string;
	    free_bytes?: number;
	    error?: string;
	
	    static createFrom(source: any = {}) {
	        return new ModelsDisk(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.drive = source["drive"];
	        this.path = source["path"];
	        this.free_bytes = source["free_bytes"];
	        this.error = source["error"];
	    }
	}
	export class NPUChipPower {
	    state: string;
	    w_mean?: number;
	    window_s: number;
	
	    static createFrom(source: any = {}) {
	        return new NPUChipPower(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.state = source["state"];
	        this.w_mean = source["w_mean"];
	        this.window_s = source["window_s"];
	    }
	}
	export class NPUMemory {
	    min_available_mib: number;
	    max_commit_pct: number;
	
	    static createFrom(source: any = {}) {
	        return new NPUMemory(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.min_available_mib = source["min_available_mib"];
	        this.max_commit_pct = source["max_commit_pct"];
	    }
	}
	export class NPUPowerScheme {
	    guid: string;
	    name: string;
	
	    static createFrom(source: any = {}) {
	        return new NPUPowerScheme(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.guid = source["guid"];
	        this.name = source["name"];
	    }
	}
	export class NPUPrefillRuntime {
	    state: string;
	    reason?: string;
	    trips: number;
	    hal_faults: number;
	    whea_events: number;
	    invalid_chunks: number;
	    chip_power: NPUChipPower;
	    memory?: NPUMemory;
	    scheme?: NPUPowerScheme;
	
	    static createFrom(source: any = {}) {
	        return new NPUPrefillRuntime(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.state = source["state"];
	        this.reason = source["reason"];
	        this.trips = source["trips"];
	        this.hal_faults = source["hal_faults"];
	        this.whea_events = source["whea_events"];
	        this.invalid_chunks = source["invalid_chunks"];
	        this.chip_power = this.convertValues(source["chip_power"], NPUChipPower);
	        this.memory = this.convertValues(source["memory"], NPUMemory);
	        this.scheme = this.convertValues(source["scheme"], NPUPowerScheme);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	
	export class Notice {
	    code: string;
	    detail: string;
	
	    static createFrom(source: any = {}) {
	        return new Notice(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.code = source["code"];
	        this.detail = source["detail"];
	    }
	}
	
	export class PendingChange {
	    key: string;
	    label: string;
	    running: string;
	    next: string;
	
	    static createFrom(source: any = {}) {
	        return new PendingChange(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.key = source["key"];
	        this.label = source["label"];
	        this.running = source["running"];
	        this.next = source["next"];
	    }
	}
	export class PreferenceValues {
	    port?: number;
	    profile?: string;
	    context?: number;
	    sessions?: number;
	    runtime_path?: string;
	    model_path?: string;
	    data_folder?: string;
	    manifest_path?: string;
	    npu_prefill?: string;
	    effort?: string;
	    start_gui_at_login?: boolean;
	    start_server_with_gui?: boolean;
	    close_to_tray?: boolean;
	    inference_expanded?: boolean;
	    allow_tight_setup?: boolean;
	    lan_access?: boolean;
	    stall_seconds?: number;
	    check_for_updates?: boolean;
	
	    static createFrom(source: any = {}) {
	        return new PreferenceValues(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.port = source["port"];
	        this.profile = source["profile"];
	        this.context = source["context"];
	        this.sessions = source["sessions"];
	        this.runtime_path = source["runtime_path"];
	        this.model_path = source["model_path"];
	        this.data_folder = source["data_folder"];
	        this.manifest_path = source["manifest_path"];
	        this.npu_prefill = source["npu_prefill"];
	        this.effort = source["effort"];
	        this.start_gui_at_login = source["start_gui_at_login"];
	        this.start_server_with_gui = source["start_server_with_gui"];
	        this.close_to_tray = source["close_to_tray"];
	        this.inference_expanded = source["inference_expanded"];
	        this.allow_tight_setup = source["allow_tight_setup"];
	        this.lan_access = source["lan_access"];
	        this.stall_seconds = source["stall_seconds"];
	        this.check_for_updates = source["check_for_updates"];
	    }
	}
	export class SwitchOverride {
	    name: string;
	    value: string;
	
	    static createFrom(source: any = {}) {
	        return new SwitchOverride(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.name = source["name"];
	        this.value = source["value"];
	    }
	}
	export class PreferencesPatch {
	    set: PreferenceValues;
	    clear: string[];
	    switch_values?: SwitchOverride[];
	
	    static createFrom(source: any = {}) {
	        return new PreferencesPatch(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.set = this.convertValues(source["set"], PreferenceValues);
	        this.clear = source["clear"];
	        this.switch_values = this.convertValues(source["switch_values"], SwitchOverride);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class PreparedCapabilities {
	    context_max?: number;
	    mtp: boolean;
	    twin: boolean;
	    mmproj: boolean;
	
	    static createFrom(source: any = {}) {
	        return new PreparedCapabilities(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.context_max = source["context_max"];
	        this.mtp = source["mtp"];
	        this.twin = source["twin"];
	        this.mmproj = source["mmproj"];
	    }
	}
	export class PreparedModel {
	    id: string;
	    display_name: string;
	    manifest_path?: string;
	    model_path?: string;
	    size_bytes: number;
	    prepared_state: string;
	    capabilities: PreparedCapabilities;
	    repair_output?: string;
	
	    static createFrom(source: any = {}) {
	        return new PreparedModel(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.display_name = source["display_name"];
	        this.manifest_path = source["manifest_path"];
	        this.model_path = source["model_path"];
	        this.size_bytes = source["size_bytes"];
	        this.prepared_state = source["prepared_state"];
	        this.capabilities = this.convertValues(source["capabilities"], PreparedCapabilities);
	        this.repair_output = source["repair_output"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class Product {
	    name: string;
	    version: string;
	    revision: string;
	    engine_version: string;
	
	    static createFrom(source: any = {}) {
	        return new Product(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.name = source["name"];
	        this.version = source["version"];
	        this.revision = source["revision"];
	        this.engine_version = source["engine_version"];
	    }
	}
	
	export class Progress {
	    component: string;
	    phase: string;
	    file_id: string;
	    bytes_done: number;
	    bytes_total?: number;
	    resume_offset: number;
	    verification_method: string;
	    overall?: JobProgress;
	
	    static createFrom(source: any = {}) {
	        return new Progress(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.component = source["component"];
	        this.phase = source["phase"];
	        this.file_id = source["file_id"];
	        this.bytes_done = source["bytes_done"];
	        this.bytes_total = source["bytes_total"];
	        this.resume_offset = source["resume_offset"];
	        this.verification_method = source["verification_method"];
	        this.overall = this.convertValues(source["overall"], JobProgress);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class Rates {
	    prefill_tokens_per_second?: number;
	    decode_tokens_per_second?: number;
	    interval_ms?: number;
	    source: string;
	    label: string;
	    unit: string;
	
	    static createFrom(source: any = {}) {
	        return new Rates(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.prefill_tokens_per_second = source["prefill_tokens_per_second"];
	        this.decode_tokens_per_second = source["decode_tokens_per_second"];
	        this.interval_ms = source["interval_ms"];
	        this.source = source["source"];
	        this.label = source["label"];
	        this.unit = source["unit"];
	    }
	}
	export class Readiness {
	    runtime: boolean;
	    model: boolean;
	    server?: boolean;
	
	    static createFrom(source: any = {}) {
	        return new Readiness(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.runtime = source["runtime"];
	        this.model = source["model"];
	        this.server = source["server"];
	    }
	}
	
	
	export class RequestRefusal {
	    code: string;
	    needed_bytes?: number;
	    free_bytes?: number;
	
	    static createFrom(source: any = {}) {
	        return new RequestRefusal(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.code = source["code"];
	        this.needed_bytes = source["needed_bytes"];
	        this.free_bytes = source["free_bytes"];
	    }
	}
	export class RequestRow {
	    id: string;
	    time: string;
	    slot?: number;
	    prompt_tokens?: number;
	    completion_tokens?: number;
	    ttft_ms?: number;
	    prefill_tps?: number;
	    decode_tps?: number;
	    cached_tokens?: number;
	    cache_kind?: string;
	    finish_reason?: string;
	    outcome: string;
	    status?: number;
	    error_code?: string;
	    refusal?: RequestRefusal;
	
	    static createFrom(source: any = {}) {
	        return new RequestRow(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.time = source["time"];
	        this.slot = source["slot"];
	        this.prompt_tokens = source["prompt_tokens"];
	        this.completion_tokens = source["completion_tokens"];
	        this.ttft_ms = source["ttft_ms"];
	        this.prefill_tps = source["prefill_tps"];
	        this.decode_tps = source["decode_tps"];
	        this.cached_tokens = source["cached_tokens"];
	        this.cache_kind = source["cache_kind"];
	        this.finish_reason = source["finish_reason"];
	        this.outcome = source["outcome"];
	        this.status = source["status"];
	        this.error_code = source["error_code"];
	        this.refusal = this.convertValues(source["refusal"], RequestRefusal);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class RuntimeView {
	    paths: Record<string, Value>;
	    runtime_kind: string;
	    readiness: Readiness;
	
	    static createFrom(source: any = {}) {
	        return new RuntimeView(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.paths = this.convertValues(source["paths"], Value, true);
	        this.runtime_kind = source["runtime_kind"];
	        this.readiness = this.convertValues(source["readiness"], Readiness);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class SavedSetupEstimate {
	    memory_estimate?: ViewMemoryEstimate;
	    reason?: string;
	    error?: CLIError;
	
	    static createFrom(source: any = {}) {
	        return new SavedSetupEstimate(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.memory_estimate = this.convertValues(source["memory_estimate"], ViewMemoryEstimate);
	        this.reason = source["reason"];
	        this.error = this.convertValues(source["error"], CLIError);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ScanWarning {
	    folder: string;
	    reason: string;
	
	    static createFrom(source: any = {}) {
	        return new ScanWarning(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.folder = source["folder"];
	        this.reason = source["reason"];
	    }
	}
	export class ServerWarning {
	    code: string;
	    remedy: string;
	
	    static createFrom(source: any = {}) {
	        return new ServerWarning(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.code = source["code"];
	        this.remedy = source["remedy"];
	    }
	}
	export class StatusFeed {
	    state: string;
	    sample_time: string;
	    age_seconds?: number;
	    error?: CLIError;
	
	    static createFrom(source: any = {}) {
	        return new StatusFeed(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.state = source["state"];
	        this.sample_time = source["sample_time"];
	        this.age_seconds = source["age_seconds"];
	        this.error = this.convertValues(source["error"], CLIError);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	
	export class StringResult {
	    value: string;
	    error?: CLIError;
	
	    static createFrom(source: any = {}) {
	        return new StringResult(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.value = source["value"];
	        this.error = this.convertValues(source["error"], CLIError);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class SuggestedFolder {
	    label: string;
	    path: string;
	    exists: boolean;
	    added: boolean;
	
	    static createFrom(source: any = {}) {
	        return new SuggestedFolder(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.label = source["label"];
	        this.path = source["path"];
	        this.exists = source["exists"];
	        this.added = source["added"];
	    }
	}
	
	export class SwitchOverrideView {
	    write_via?: string;
	    preference_field?: string;
	    preference_type?: string;
	    name: string;
	    saved?: string;
	    effective: Value;
	    write_block: string;
	    restart_state: string;
	
	    static createFrom(source: any = {}) {
	        return new SwitchOverrideView(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.write_via = source["write_via"];
	        this.preference_field = source["preference_field"];
	        this.preference_type = source["preference_type"];
	        this.name = source["name"];
	        this.saved = source["saved"];
	        this.effective = this.convertValues(source["effective"], Value);
	        this.write_block = source["write_block"];
	        this.restart_state = source["restart_state"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class SwitchOverridesPatch {
	    set: SwitchOverride[];
	    clear: string[];
	
	    static createFrom(source: any = {}) {
	        return new SwitchOverridesPatch(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.set = this.convertValues(source["set"], SwitchOverride);
	        this.clear = source["clear"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class UpdateProgress {
	    phase: string;
	    bytes_done: number;
	    bytes_total?: number;
	
	    static createFrom(source: any = {}) {
	        return new UpdateProgress(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.phase = source["phase"];
	        this.bytes_done = source["bytes_done"];
	        this.bytes_total = source["bytes_total"];
	    }
	}
	export class Update {
	    state: string;
	    ready: boolean;
	    runtime_followup_required: boolean;
	    log_path: string;
	    log_lines: string[];
	    current_version: string;
	    available_version?: string;
	    notes?: string;
	    size_bytes?: number;
	    runtime_download_bytes?: number;
	    models_download_bytes?: number;
	    progress?: UpdateProgress;
	    error?: CLIError;
	    last_result_notice?: string;
	    check_failed: boolean;
	    previous_version?: string;
	
	    static createFrom(source: any = {}) {
	        return new Update(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.state = source["state"];
	        this.ready = source["ready"];
	        this.runtime_followup_required = source["runtime_followup_required"];
	        this.log_path = source["log_path"];
	        this.log_lines = source["log_lines"];
	        this.current_version = source["current_version"];
	        this.available_version = source["available_version"];
	        this.notes = source["notes"];
	        this.size_bytes = source["size_bytes"];
	        this.runtime_download_bytes = source["runtime_download_bytes"];
	        this.models_download_bytes = source["models_download_bytes"];
	        this.progress = this.convertValues(source["progress"], UpdateProgress);
	        this.error = this.convertValues(source["error"], CLIError);
	        this.last_result_notice = source["last_result_notice"];
	        this.check_failed = source["check_failed"];
	        this.previous_version = source["previous_version"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	
	export class UserOverrides {
	    port?: number;
	    profile?: string;
	    context?: number;
	    sessions?: number;
	    runtime_path?: string;
	    model_path?: string;
	    data_folder?: string;
	    manifest_path?: string;
	    npu_prefill?: string;
	    effort?: string;
	    model_folders: string[];
	    start_gui_at_login: boolean;
	    start_server_with_gui: boolean;
	    close_to_tray: boolean;
	    inference_expanded: boolean;
	    allow_tight_setup: boolean;
	    lan_access: boolean;
	    stall_seconds?: number;
	    check_for_updates?: boolean;
	    switch_overrides?: Record<string, string>;
	
	    static createFrom(source: any = {}) {
	        return new UserOverrides(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.port = source["port"];
	        this.profile = source["profile"];
	        this.context = source["context"];
	        this.sessions = source["sessions"];
	        this.runtime_path = source["runtime_path"];
	        this.model_path = source["model_path"];
	        this.data_folder = source["data_folder"];
	        this.manifest_path = source["manifest_path"];
	        this.npu_prefill = source["npu_prefill"];
	        this.effort = source["effort"];
	        this.model_folders = source["model_folders"];
	        this.start_gui_at_login = source["start_gui_at_login"];
	        this.start_server_with_gui = source["start_server_with_gui"];
	        this.close_to_tray = source["close_to_tray"];
	        this.inference_expanded = source["inference_expanded"];
	        this.allow_tight_setup = source["allow_tight_setup"];
	        this.lan_access = source["lan_access"];
	        this.stall_seconds = source["stall_seconds"];
	        this.check_for_updates = source["check_for_updates"];
	        this.switch_overrides = source["switch_overrides"];
	    }
	}
	
	export class ViewLoopBreaker {
	    value: boolean;
	    source: string;
	    requested: Value;
	
	    static createFrom(source: any = {}) {
	        return new ViewLoopBreaker(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.value = source["value"];
	        this.source = source["source"];
	        this.requested = this.convertValues(source["requested"], Value);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ViewThinkMarkerBias {
	    value: number;
	    source: string;
	    requested: Value;
	    by_effort: EffortBiases;
	
	    static createFrom(source: any = {}) {
	        return new ViewThinkMarkerBias(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.value = source["value"];
	        this.source = source["source"];
	        this.requested = this.convertValues(source["requested"], Value);
	        this.by_effort = this.convertValues(source["by_effort"], EffortBiases);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ViewStall {
	    seconds: number;
	    threshold_seconds: number;
	    since: string;
	    message: string;
	
	    static createFrom(source: any = {}) {
	        return new ViewStall(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.seconds = source["seconds"];
	        this.threshold_seconds = source["threshold_seconds"];
	        this.since = source["since"];
	        this.message = source["message"];
	    }
	}
	export class WindowView {
	    tray_action: string;
	    tray_available: boolean;
	    closing: boolean;
	
	    static createFrom(source: any = {}) {
	        return new WindowView(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.tray_action = source["tray_action"];
	        this.tray_available = source["tray_available"];
	        this.closing = source["closing"];
	    }
	}
	export class ViewScannedModel {
	    path: string;
	    file_name: string;
	    folder: string;
	    size_bytes: number;
	    shards_found: number;
	    shards_expected: number;
	    architecture?: string;
	    model_name?: string;
	    native_context_tokens?: number;
	    kind: string;
	    runnable: boolean;
	    reason?: string;
	    route?: string;
	    estimate: ViewModelEstimateInfo;
	
	    static createFrom(source: any = {}) {
	        return new ViewScannedModel(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.path = source["path"];
	        this.file_name = source["file_name"];
	        this.folder = source["folder"];
	        this.size_bytes = source["size_bytes"];
	        this.shards_found = source["shards_found"];
	        this.shards_expected = source["shards_expected"];
	        this.architecture = source["architecture"];
	        this.model_name = source["model_name"];
	        this.native_context_tokens = source["native_context_tokens"];
	        this.kind = source["kind"];
	        this.runnable = source["runnable"];
	        this.reason = source["reason"];
	        this.route = source["route"];
	        this.estimate = this.convertValues(source["estimate"], ViewModelEstimateInfo);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ViewModelEstimateInfo {
	    estimate_state: string;
	    reason?: string;
	    evidence_label?: string;
	    fit_verified: boolean;
	    fit_inputs: Record<string, Value>;
	    notes: string[];
	    memory_estimate?: ViewMemoryEstimate;
	
	    static createFrom(source: any = {}) {
	        return new ViewModelEstimateInfo(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.estimate_state = source["estimate_state"];
	        this.reason = source["reason"];
	        this.evidence_label = source["evidence_label"];
	        this.fit_verified = source["fit_verified"];
	        this.fit_inputs = this.convertValues(source["fit_inputs"], Value, true);
	        this.notes = source["notes"];
	        this.memory_estimate = this.convertValues(source["memory_estimate"], ViewMemoryEstimate);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ViewCatalogueEntry {
	    id: string;
	    display_name: string;
	    technical_name: string;
	    about: string;
	    selection_label: string;
	    about_links: string[];
	    gpu_bytes?: number;
	    system_memory_bytes?: number;
	    system_memory_note?: string;
	    size_split_source?: string;
	    role: string;
	    architecture: string;
	    set?: string;
	    default_status: string;
	    family_group?: string;
	    download_enabled: boolean;
	    disabled_reason?: string;
	    packaging: string;
	    licence: ModelLicence;
	    files: ModelFile[];
	    download_bytes: number;
	    prepared_state: string;
	    local_folder: string;
	    estimate: ViewModelEstimateInfo;
	
	    static createFrom(source: any = {}) {
	        return new ViewCatalogueEntry(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.display_name = source["display_name"];
	        this.technical_name = source["technical_name"];
	        this.about = source["about"];
	        this.selection_label = source["selection_label"];
	        this.about_links = source["about_links"];
	        this.gpu_bytes = source["gpu_bytes"];
	        this.system_memory_bytes = source["system_memory_bytes"];
	        this.system_memory_note = source["system_memory_note"];
	        this.size_split_source = source["size_split_source"];
	        this.role = source["role"];
	        this.architecture = source["architecture"];
	        this.set = source["set"];
	        this.default_status = source["default_status"];
	        this.family_group = source["family_group"];
	        this.download_enabled = source["download_enabled"];
	        this.disabled_reason = source["disabled_reason"];
	        this.packaging = source["packaging"];
	        this.licence = this.convertValues(source["licence"], ModelLicence);
	        this.files = this.convertValues(source["files"], ModelFile);
	        this.download_bytes = source["download_bytes"];
	        this.prepared_state = source["prepared_state"];
	        this.local_folder = source["local_folder"];
	        this.estimate = this.convertValues(source["estimate"], ViewModelEstimateInfo);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ViewModelsListing {
	    models_root: string;
	    available_bytes?: number;
	    scan_depth_limit: number;
	    estimate_defaults: Value;
	    catalogue_version: string;
	    package_files_skipped: number;
	    scanned_folders: string[];
	
	    static createFrom(source: any = {}) {
	        return new ViewModelsListing(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.models_root = source["models_root"];
	        this.available_bytes = source["available_bytes"];
	        this.scan_depth_limit = source["scan_depth_limit"];
	        this.estimate_defaults = this.convertValues(source["estimate_defaults"], Value);
	        this.catalogue_version = source["catalogue_version"];
	        this.package_files_skipped = source["package_files_skipped"];
	        this.scanned_folders = source["scanned_folders"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ViewModelsState {
	    status: string;
	    reason?: string;
	    reason_code?: string;
	    exit_code?: number;
	    updated_at?: string;
	    refreshing: boolean;
	    listing?: ViewModelsListing;
	    catalogue: ViewCatalogueEntry[];
	    prepared_models: PreparedModel[];
	    scanned: ViewScannedModel[];
	    scan_warnings: ScanWarning[];
	    disk: ModelsDisk;
	    suggested_folders: SuggestedFolder[];
	    folders: string[];
	    download?: ModelJob;
	
	    static createFrom(source: any = {}) {
	        return new ViewModelsState(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.status = source["status"];
	        this.reason = source["reason"];
	        this.reason_code = source["reason_code"];
	        this.exit_code = source["exit_code"];
	        this.updated_at = source["updated_at"];
	        this.refreshing = source["refreshing"];
	        this.listing = this.convertValues(source["listing"], ViewModelsListing);
	        this.catalogue = this.convertValues(source["catalogue"], ViewCatalogueEntry);
	        this.prepared_models = this.convertValues(source["prepared_models"], PreparedModel);
	        this.scanned = this.convertValues(source["scanned"], ViewScannedModel);
	        this.scan_warnings = this.convertValues(source["scan_warnings"], ScanWarning);
	        this.disk = this.convertValues(source["disk"], ModelsDisk);
	        this.suggested_folders = this.convertValues(source["suggested_folders"], SuggestedFolder);
	        this.folders = source["folders"];
	        this.download = this.convertValues(source["download"], ModelJob);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ViewGPU {
	    sample_time?: string;
	    age_seconds?: number;
	    window_seconds?: number;
	    use_pct?: number;
	    memory_used_bytes?: number;
	    memory_total_bytes?: number;
	    shared_used_bytes?: number;
	    temperature_c?: number;
	    chip_power_w?: number;
	
	    static createFrom(source: any = {}) {
	        return new ViewGPU(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.sample_time = source["sample_time"];
	        this.age_seconds = source["age_seconds"];
	        this.window_seconds = source["window_seconds"];
	        this.use_pct = source["use_pct"];
	        this.memory_used_bytes = source["memory_used_bytes"];
	        this.memory_total_bytes = source["memory_total_bytes"];
	        this.shared_used_bytes = source["shared_used_bytes"];
	        this.temperature_c = source["temperature_c"];
	        this.chip_power_w = source["chip_power_w"];
	    }
	}
	export class ViewSlot {
	    id: number;
	    processing: boolean;
	    state: string;
	    state_label: string;
	    tokens_used?: number;
	    speculative: boolean;
	    request_id?: number;
	    context_tokens?: number;
	    prompt_tokens?: number;
	    cached_prompt_tokens?: number;
	    prefilled_tokens?: number;
	    generated_tokens?: number;
	    remaining_tokens?: number;
	    prefill_tps?: number;
	    decode_tps?: number;
	    elapsed_ms?: number;
	
	    static createFrom(source: any = {}) {
	        return new ViewSlot(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.processing = source["processing"];
	        this.state = source["state"];
	        this.state_label = source["state_label"];
	        this.tokens_used = source["tokens_used"];
	        this.speculative = source["speculative"];
	        this.request_id = source["request_id"];
	        this.context_tokens = source["context_tokens"];
	        this.prompt_tokens = source["prompt_tokens"];
	        this.cached_prompt_tokens = source["cached_prompt_tokens"];
	        this.prefilled_tokens = source["prefilled_tokens"];
	        this.generated_tokens = source["generated_tokens"];
	        this.remaining_tokens = source["remaining_tokens"];
	        this.prefill_tps = source["prefill_tps"];
	        this.decode_tps = source["decode_tps"];
	        this.elapsed_ms = source["elapsed_ms"];
	    }
	}
	export class ViewEffectiveSwitch {
	    name: string;
	    label?: string;
	    effective: Value;
	    source: string;
	    reason?: string;
	    audience: string;
	    group: string;
	
	    static createFrom(source: any = {}) {
	        return new ViewEffectiveSwitch(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.name = source["name"];
	        this.label = source["label"];
	        this.effective = this.convertValues(source["effective"], Value);
	        this.source = source["source"];
	        this.reason = source["reason"];
	        this.audience = source["audience"];
	        this.group = source["group"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ViewSwitch {
	    name: string;
	    label?: string;
	    kind: string;
	    class: string;
	    model_family: string;
	    platform: string;
	    requires: string[];
	    values: Record<string, Value>;
	    exact_off: boolean;
	    inject_with?: string;
	    suppressed_by: string[];
	    purpose: string;
	    default_text: string;
	    hardware_scope: string;
	    note: string;
	    built_in?: string;
	    value_rule: string;
	    aliases: string[];
	    takes_value: boolean;
	    echo: boolean;
	    secret?: boolean;
	    group: string;
	    audience: string;
	    experimental: boolean;
	    restart_required: boolean;
	    gui_control: boolean;
	    allowed: Value;
	
	    static createFrom(source: any = {}) {
	        return new ViewSwitch(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.name = source["name"];
	        this.label = source["label"];
	        this.kind = source["kind"];
	        this.class = source["class"];
	        this.model_family = source["model_family"];
	        this.platform = source["platform"];
	        this.requires = source["requires"];
	        this.values = this.convertValues(source["values"], Value, true);
	        this.exact_off = source["exact_off"];
	        this.inject_with = source["inject_with"];
	        this.suppressed_by = source["suppressed_by"];
	        this.purpose = source["purpose"];
	        this.default_text = source["default_text"];
	        this.hardware_scope = source["hardware_scope"];
	        this.note = source["note"];
	        this.built_in = source["built_in"];
	        this.value_rule = source["value_rule"];
	        this.aliases = source["aliases"];
	        this.takes_value = source["takes_value"];
	        this.echo = source["echo"];
	        this.secret = source["secret"];
	        this.group = source["group"];
	        this.audience = source["audience"];
	        this.experimental = source["experimental"];
	        this.restart_required = source["restart_required"];
	        this.gui_control = source["gui_control"];
	        this.allowed = this.convertValues(source["allowed"], Value);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ViewSettingRow {
	    name: string;
	    requested: Value;
	    effective: Value;
	    source: string;
	    reason?: string;
	    allowed?: Value;
	
	    static createFrom(source: any = {}) {
	        return new ViewSettingRow(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.name = source["name"];
	        this.requested = this.convertValues(source["requested"], Value);
	        this.effective = this.convertValues(source["effective"], Value);
	        this.source = source["source"];
	        this.reason = source["reason"];
	        this.allowed = this.convertValues(source["allowed"], Value);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ViewFailure {
	    code: string;
	    source: string;
	    time: string;
	    remedy: string;
	    detail?: string;
	    memory?: MemoryShortfall;
	    again?: boolean;
	    forced: boolean;
	    component?: string;
	    stage?: string;
	    switch?: string;
	
	    static createFrom(source: any = {}) {
	        return new ViewFailure(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.code = source["code"];
	        this.source = source["source"];
	        this.time = source["time"];
	        this.remedy = source["remedy"];
	        this.detail = source["detail"];
	        this.memory = this.convertValues(source["memory"], MemoryShortfall);
	        this.again = source["again"];
	        this.forced = source["forced"];
	        this.component = source["component"];
	        this.stage = source["stage"];
	        this.switch = source["switch"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class ViewInstance {
	    active_model?: ActiveModel;
	    schema_version: number;
	    control_version: number;
	    instance_id: string;
	    pid: number;
	    process_creation_time: string;
	    exe_path: string;
	    identity?: Identity;
	    phase: string;
	    endpoint?: string;
	    log_path?: string;
	    last_error?: CLIError;
	    ready: boolean;
	    effective_settings: Record<string, Value>;
	    served_model: string;
	
	    static createFrom(source: any = {}) {
	        return new ViewInstance(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.active_model = this.convertValues(source["active_model"], ActiveModel);
	        this.schema_version = source["schema_version"];
	        this.control_version = source["control_version"];
	        this.instance_id = source["instance_id"];
	        this.pid = source["pid"];
	        this.process_creation_time = source["process_creation_time"];
	        this.exe_path = source["exe_path"];
	        this.identity = this.convertValues(source["identity"], Identity);
	        this.phase = source["phase"];
	        this.endpoint = source["endpoint"];
	        this.log_path = source["log_path"];
	        this.last_error = this.convertValues(source["last_error"], CLIError);
	        this.ready = source["ready"];
	        this.effective_settings = this.convertValues(source["effective_settings"], Value, true);
	        this.served_model = source["served_model"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class View {
	    preparation_file?: FilePosition;
	    actions: LifecycleActions;
	    status_label: string;
	    serving: boolean;
	    auto_restarted: boolean;
	    server_warnings: ServerWarning[];
	    owned_server: boolean;
	    current_model?: CurrentModel;
	    fit: FitVerdict;
	    _source?: string;
	    state: string;
	    reason: string;
	    remedy: string;
	    initialized: boolean;
	    startup_phase: string;
	    prepared: boolean;
	    stale: boolean;
	    generation: number;
	    managed: boolean;
	    management_supported: boolean;
	    management_message: string;
	    phase: string;
	    instance?: ViewInstance;
	    discovered: ViewInstance[];
	    failure?: ViewFailure;
	    settings: ViewSettingRow[];
	    switches: ViewSwitch[];
	    switch_overrides: SwitchOverrideView[];
	    effective_switches: ViewEffectiveSwitch[];
	    profiles: Profile[];
	    product?: Product;
	    capabilities?: Capabilities;
	    memory_estimate?: ViewMemoryEstimate;
	    saved_estimate: SavedSetupEstimate;
	    memory_estimate_reason?: string;
	    process_memory?: MemoryUse;
	    connection?: Connection;
	    log_path?: string;
	    active_model?: ActiveModel;
	    client_snippets: ClientSnippet[];
	    npu_prefill_runtime?: NPUPrefillRuntime;
	    endpoints?: Endpoints;
	    api_key_env: string;
	    pi_snippet: string;
	    rates?: Rates;
	    sessions?: number;
	    queue?: number;
	    slots: ViewSlot[];
	    live_samples: LiveSample[];
	    live_sample_time: string;
	    requests: RequestRow[];
	    counters?: Counters;
	    last_completed_request?: LastCompletedRequest;
	    gpu?: ViewGPU;
	    models?: ViewModelsState;
	    prepared_models: PreparedModel[];
	    prepared_default?: string;
	    runtime: RuntimeView;
	    saved: UserOverrides;
	    download_folder: DownloadFolderView;
	    key: KeyView;
	    window: WindowView;
	    stall?: ViewStall;
	    progress?: Progress;
	    approval?: Approval;
	    preparation: ModelPlan[];
	    preparation_pending: boolean;
	    plan_error?: CLIError;
	    restart_needed: PendingChange[];
	    inference_defaults?: InferenceDefaults;
	    think_marker_bias?: ViewThinkMarkerBias;
	    loop_breaker?: ViewLoopBreaker;
	    think_marker_entropy_gate?: EntropyGate;
	    logs: string[];
	    notice?: Notice;
	    warnings: string[];
	    diagnostic: string;
	    bundle_preview: string;
	    update: Update;
	    check_for_updates: boolean;
	    status_feed: StatusFeed;
	
	    static createFrom(source: any = {}) {
	        return new View(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.preparation_file = this.convertValues(source["preparation_file"], FilePosition);
	        this.actions = this.convertValues(source["actions"], LifecycleActions);
	        this.status_label = source["status_label"];
	        this.serving = source["serving"];
	        this.auto_restarted = source["auto_restarted"];
	        this.server_warnings = this.convertValues(source["server_warnings"], ServerWarning);
	        this.owned_server = source["owned_server"];
	        this.current_model = this.convertValues(source["current_model"], CurrentModel);
	        this.fit = this.convertValues(source["fit"], FitVerdict);
	        this._source = source["_source"];
	        this.state = source["state"];
	        this.reason = source["reason"];
	        this.remedy = source["remedy"];
	        this.initialized = source["initialized"];
	        this.startup_phase = source["startup_phase"];
	        this.prepared = source["prepared"];
	        this.stale = source["stale"];
	        this.generation = source["generation"];
	        this.managed = source["managed"];
	        this.management_supported = source["management_supported"];
	        this.management_message = source["management_message"];
	        this.phase = source["phase"];
	        this.instance = this.convertValues(source["instance"], ViewInstance);
	        this.discovered = this.convertValues(source["discovered"], ViewInstance);
	        this.failure = this.convertValues(source["failure"], ViewFailure);
	        this.settings = this.convertValues(source["settings"], ViewSettingRow);
	        this.switches = this.convertValues(source["switches"], ViewSwitch);
	        this.switch_overrides = this.convertValues(source["switch_overrides"], SwitchOverrideView);
	        this.effective_switches = this.convertValues(source["effective_switches"], ViewEffectiveSwitch);
	        this.profiles = this.convertValues(source["profiles"], Profile);
	        this.product = this.convertValues(source["product"], Product);
	        this.capabilities = this.convertValues(source["capabilities"], Capabilities);
	        this.memory_estimate = this.convertValues(source["memory_estimate"], ViewMemoryEstimate);
	        this.saved_estimate = this.convertValues(source["saved_estimate"], SavedSetupEstimate);
	        this.memory_estimate_reason = source["memory_estimate_reason"];
	        this.process_memory = this.convertValues(source["process_memory"], MemoryUse);
	        this.connection = this.convertValues(source["connection"], Connection);
	        this.log_path = source["log_path"];
	        this.active_model = this.convertValues(source["active_model"], ActiveModel);
	        this.client_snippets = this.convertValues(source["client_snippets"], ClientSnippet);
	        this.npu_prefill_runtime = this.convertValues(source["npu_prefill_runtime"], NPUPrefillRuntime);
	        this.endpoints = this.convertValues(source["endpoints"], Endpoints);
	        this.api_key_env = source["api_key_env"];
	        this.pi_snippet = source["pi_snippet"];
	        this.rates = this.convertValues(source["rates"], Rates);
	        this.sessions = source["sessions"];
	        this.queue = source["queue"];
	        this.slots = this.convertValues(source["slots"], ViewSlot);
	        this.live_samples = this.convertValues(source["live_samples"], LiveSample);
	        this.live_sample_time = source["live_sample_time"];
	        this.requests = this.convertValues(source["requests"], RequestRow);
	        this.counters = this.convertValues(source["counters"], Counters);
	        this.last_completed_request = this.convertValues(source["last_completed_request"], LastCompletedRequest);
	        this.gpu = this.convertValues(source["gpu"], ViewGPU);
	        this.models = this.convertValues(source["models"], ViewModelsState);
	        this.prepared_models = this.convertValues(source["prepared_models"], PreparedModel);
	        this.prepared_default = source["prepared_default"];
	        this.runtime = this.convertValues(source["runtime"], RuntimeView);
	        this.saved = this.convertValues(source["saved"], UserOverrides);
	        this.download_folder = this.convertValues(source["download_folder"], DownloadFolderView);
	        this.key = this.convertValues(source["key"], KeyView);
	        this.window = this.convertValues(source["window"], WindowView);
	        this.stall = this.convertValues(source["stall"], ViewStall);
	        this.progress = this.convertValues(source["progress"], Progress);
	        this.approval = this.convertValues(source["approval"], Approval);
	        this.preparation = this.convertValues(source["preparation"], ModelPlan);
	        this.preparation_pending = source["preparation_pending"];
	        this.plan_error = this.convertValues(source["plan_error"], CLIError);
	        this.restart_needed = this.convertValues(source["restart_needed"], PendingChange);
	        this.inference_defaults = this.convertValues(source["inference_defaults"], InferenceDefaults);
	        this.think_marker_bias = this.convertValues(source["think_marker_bias"], ViewThinkMarkerBias);
	        this.loop_breaker = this.convertValues(source["loop_breaker"], ViewLoopBreaker);
	        this.think_marker_entropy_gate = this.convertValues(source["think_marker_entropy_gate"], EntropyGate);
	        this.logs = source["logs"];
	        this.notice = this.convertValues(source["notice"], Notice);
	        this.warnings = source["warnings"];
	        this.diagnostic = source["diagnostic"];
	        this.bundle_preview = source["bundle_preview"];
	        this.update = this.convertValues(source["update"], Update);
	        this.check_for_updates = source["check_for_updates"];
	        this.status_feed = this.convertValues(source["status_feed"], StatusFeed);
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

}

