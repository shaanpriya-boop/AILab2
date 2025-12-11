"""
Enterprise Agentic Observability Platform
Core observability engine - DO NOT confuse with langgraphapp.py!
"""

import time
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import threading
import logging
from contextlib import contextmanager
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of observable events"""
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    GUARDRAIL_CHECK = "guardrail_check"
    HALLUCINATION_CHECK = "hallucination_check"
    ERROR = "error"
    MEMORY_UPDATE = "memory_update"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"


class Severity(Enum):
    """Event severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TraceEvent:
    """Individual trace event"""
    event_id: str
    trace_id: str
    parent_id: Optional[str]
    event_type: EventType
    timestamp: float
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    severity: Severity = Severity.INFO
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'event_type': self.event_type.value,
            'severity': self.severity.value
        }


@dataclass
class GuardrailViolation:
    """Guardrail violation record"""
    rule_name: str
    violation_type: str
    severity: Severity
    content: str
    timestamp: float
    remediation: Optional[str] = None


@dataclass
class HallucinationDetection:
    """Hallucination detection result"""
    confidence: float
    detected: bool
    evidence: List[str]
    timestamp: float
    model_response: str
    context_used: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: float
    latency_ms: float
    token_count: int
    cost_usd: float
    memory_mb: float
    success: bool


class ObservabilityCollector:
    """Central collector for all observability data"""
    
    def __init__(self):
        self.traces: Dict[str, List[TraceEvent]] = defaultdict(list)
        self.guardrail_violations: List[GuardrailViolation] = []
        self.hallucinations: List[HallucinationDetection] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        self.anomalies: List[Dict[str, Any]] = []
        
        self.token_usage: Dict[str, int] = defaultdict(int)
        self.cost_tracking: Dict[str, float] = defaultdict(float)
        self.latency_tracking: List[float] = []
        
        self._lock = threading.Lock()
    
    def record_event(self, event: TraceEvent):
        with self._lock:
            self.traces[event.trace_id].append(event)
            logger.debug(f"Recorded event: {event.event_type.value} - {event.name}")
    
    def record_guardrail_violation(self, violation: GuardrailViolation):
        with self._lock:
            self.guardrail_violations.append(violation)
            logger.warning(f"Guardrail violation: {violation.rule_name}")
    
    def record_hallucination(self, detection: HallucinationDetection):
        with self._lock:
            self.hallucinations.append(detection)
            if detection.detected:
                logger.warning(f"Hallucination detected: {detection.confidence:.2f}")
    
    def record_performance(self, metrics: PerformanceMetrics):
        with self._lock:
            self.performance_metrics.append(metrics)
            self.latency_tracking.append(metrics.latency_ms)
    
    def record_token_usage(self, model: str, tokens: int, cost: float):
        with self._lock:
            self.token_usage[model] += tokens
            self.cost_tracking[model] += cost
    
    def get_trace(self, trace_id: str) -> List[TraceEvent]:
        return self.traces.get(trace_id, [])
    
    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_traces": len(self.traces),
                "total_events": sum(len(events) for events in self.traces.values()),
                "guardrail_violations": len(self.guardrail_violations),
                "hallucinations_detected": sum(1 for h in self.hallucinations if h.detected),
                "total_tokens": sum(self.token_usage.values()),
                "total_cost_usd": sum(self.cost_tracking.values()),
                "avg_latency_ms": sum(self.latency_tracking) / len(self.latency_tracking) if self.latency_tracking else 0,
                "p95_latency_ms": self._percentile(self.latency_tracking, 95) if self.latency_tracking else 0,
                "success_rate": sum(1 for m in self.performance_metrics if m.success) / len(self.performance_metrics) * 100 if self.performance_metrics else 0
            }
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


_collector = ObservabilityCollector()


class ObservabilityContext:
    """Thread-local context for tracking traces"""
    
    def __init__(self):
        self._local = threading.local()
    
    @property
    def trace_id(self) -> Optional[str]:
        return getattr(self._local, 'trace_id', None)
    
    @trace_id.setter
    def trace_id(self, value: str):
        self._local.trace_id = value
    
    @property
    def parent_id(self) -> Optional[str]:
        return getattr(self._local, 'parent_id', None)
    
    @parent_id.setter
    def parent_id(self, value: str):
        self._local.parent_id = value


_context = ObservabilityContext()


@contextmanager
def trace_workflow(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for tracing entire workflow"""
    trace_id = str(uuid.uuid4())
    _context.trace_id = trace_id
    
    start_event = TraceEvent(
        event_id=str(uuid.uuid4()),
        trace_id=trace_id,
        parent_id=None,
        event_type=EventType.WORKFLOW_START,
        timestamp=time.time(),
        name=name,
        metadata=metadata or {}
    )
    _collector.record_event(start_event)
    
    start_time = time.time()
    success = False
    error_msg = None
    
    try:
        yield trace_id
        success = True
    except Exception as e:
        error_msg = str(e)
        error_event = TraceEvent(
            event_id=str(uuid.uuid4()),
            trace_id=trace_id,
            parent_id=None,
            event_type=EventType.ERROR,
            timestamp=time.time(),
            name=f"Error in {name}",
            metadata={"error": error_msg},
            severity=Severity.ERROR
        )
        _collector.record_event(error_event)
        raise
    finally:
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        
        end_event = TraceEvent(
            event_id=str(uuid.uuid4()),
            trace_id=trace_id,
            parent_id=None,
            event_type=EventType.WORKFLOW_END,
            timestamp=end_time,
            name=name,
            metadata={"success": success, "error": error_msg},
            metrics={"latency_ms": latency}
        )
        _collector.record_event(end_event)


def trace_agent(name: Optional[str] = None):
    """Decorator for tracing agent execution"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            agent_name = name or func.__name__
            event_id = str(uuid.uuid4())
            
            previous_parent = _context.parent_id
            _context.parent_id = event_id
            
            start_event = TraceEvent(
                event_id=event_id,
                trace_id=_context.trace_id or str(uuid.uuid4()),
                parent_id=previous_parent,
                event_type=EventType.AGENT_START,
                timestamp=time.time(),
                name=agent_name,
                metadata={"args": str(args)[:200], "kwargs": str(kwargs)[:200]}
            )
            _collector.record_event(start_event)
            
            start_time = time.time()
            success = False
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_event = TraceEvent(
                    event_id=str(uuid.uuid4()),
                    trace_id=_context.trace_id,
                    parent_id=event_id,
                    event_type=EventType.ERROR,
                    timestamp=time.time(),
                    name=f"Error in {agent_name}",
                    metadata={"error": str(e)},
                    severity=Severity.ERROR
                )
                _collector.record_event(error_event)
                raise
            finally:
                end_time = time.time()
                latency = (end_time - start_time) * 1000
                
                end_event = TraceEvent(
                    event_id=str(uuid.uuid4()),
                    trace_id=_context.trace_id,
                    parent_id=event_id,
                    event_type=EventType.AGENT_END,
                    timestamp=end_time,
                    name=agent_name,
                    metadata={"success": success},
                    metrics={"latency_ms": latency}
                )
                _collector.record_event(end_event)
                
                _context.parent_id = previous_parent
        
        return wrapper
    return decorator


def trace_llm_call(model: str, tokens: int, cost: float = 0.0):
    """Record LLM call"""
    event = TraceEvent(
        event_id=str(uuid.uuid4()),
        trace_id=_context.trace_id or "unknown",
        parent_id=_context.parent_id,
        event_type=EventType.LLM_CALL,
        timestamp=time.time(),
        name=f"LLM Call: {model}",
        metadata={"model": model},
        metrics={"tokens": tokens, "cost_usd": cost}
    )
    _collector.record_event(event)
    _collector.record_token_usage(model, tokens, cost)


def trace_tool_call(tool_name: str, input_data: Any, output_data: Any, latency_ms: float):
    """Record tool call"""
    event = TraceEvent(
        event_id=str(uuid.uuid4()),
        trace_id=_context.trace_id or "unknown",
        parent_id=_context.parent_id,
        event_type=EventType.TOOL_CALL,
        timestamp=time.time(),
        name=f"Tool: {tool_name}",
        metadata={
            "input": str(input_data)[:500],
            "output": str(output_data)[:500]
        },
        metrics={"latency_ms": latency_ms}
    )
    _collector.record_event(event)


class GuardrailSystem:
    """Guardrail checking system"""
    
    def __init__(self):
        self.rules: List[tuple] = []
    
    def add_rule(self, name: str, check_func: Callable[[str], bool], severity: Severity = Severity.WARNING):
        self.rules.append((name, check_func, severity))
    
    def check(self, content: str) -> List[GuardrailViolation]:
        violations = []
        
        for rule_name, check_func, severity in self.rules:
            try:
                if not check_func(content):
                    violation = GuardrailViolation(
                        rule_name=rule_name,
                        violation_type="policy_violation",
                        severity=severity,
                        content=content[:200],
                        timestamp=time.time()
                    )
                    violations.append(violation)
                    _collector.record_guardrail_violation(violation)
            except Exception as e:
                logger.error(f"Guardrail check failed for {rule_name}: {e}")
        
        return violations


class HallucinationDetector:
    """Hallucination detection system"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def detect(
        self,
        model_response: str,
        source_context: Optional[str] = None,
        ground_truth: Optional[str] = None
    ) -> HallucinationDetection:
        confidence = 0.0
        evidence = []
        
        if source_context:
            response_words = set(model_response.lower().split())
            context_words = set(source_context.lower().split())
            unsupported_ratio = len(response_words - context_words) / len(response_words) if response_words else 0
            
            if unsupported_ratio > 0.5:
                confidence += 0.3
                evidence.append("High ratio of unsupported claims")
        
        uncertainty_markers = ["maybe", "possibly", "might", "could be", "i think", "perhaps"]
        if any(marker in model_response.lower() for marker in uncertainty_markers):
            confidence += 0.2
            evidence.append("Contains uncertainty markers")
        
        import re
        numbers = re.findall(r'\d+', model_response)
        if len(numbers) > 5:
            confidence += 0.1
            evidence.append("Contains many specific numbers")
        
        detected = confidence >= self.threshold
        
        detection = HallucinationDetection(
            confidence=confidence,
            detected=detected,
            evidence=evidence,
            timestamp=time.time(),
            model_response=model_response[:500],
            context_used=source_context[:500] if source_context else None
        )
        
        _collector.record_hallucination(detection)
        
        return detection


class AnomalyDetector:
    """Anomaly detection for performance"""
    
    def __init__(self):
        self.baseline_latency: Optional[float] = None
        self.baseline_tokens: Optional[float] = None
        self.anomaly_threshold = 2.0
    
    def update_baseline(self, latency_ms: float, token_count: int):
        if self.baseline_latency is None:
            self.baseline_latency = latency_ms
            self.baseline_tokens = float(token_count)
        else:
            alpha = 0.1
            self.baseline_latency = alpha * latency_ms + (1 - alpha) * self.baseline_latency
            self.baseline_tokens = alpha * token_count + (1 - alpha) * self.baseline_tokens
    
    def detect_anomaly(self, latency_ms: float, token_count: int) -> Optional[Dict[str, Any]]:
        if self.baseline_latency is None:
            self.update_baseline(latency_ms, token_count)
            return None
        
        anomalies = []
        
        if latency_ms > self.baseline_latency * self.anomaly_threshold:
            anomalies.append({
                "type": "high_latency",
                "value": latency_ms,
                "baseline": self.baseline_latency,
                "ratio": latency_ms / self.baseline_latency
            })
        
        if token_count > self.baseline_tokens * self.anomaly_threshold:
            anomalies.append({
                "type": "high_token_usage",
                "value": token_count,
                "baseline": self.baseline_tokens,
                "ratio": token_count / self.baseline_tokens
            })
        
        if anomalies:
            anomaly_record = {
                "timestamp": time.time(),
                "anomalies": anomalies
            }
            _collector.anomalies.append(anomaly_record)
            logger.warning(f"Anomaly detected: {anomalies}")
            return anomaly_record
        
        self.update_baseline(latency_ms, token_count)
        return None


class ObservabilityDashboard:
    """Dashboard for viewing observability data"""
    
    def __init__(self, collector: ObservabilityCollector):
        self.collector = collector
    
    def print_summary(self):
        summary = self.collector.get_summary()
        
        print("\n" + "="*80)
        print("OBSERVABILITY DASHBOARD - SUMMARY")
        print("="*80)
        
        print(f"\n📊 EXECUTION METRICS:")
        print(f"  Total Traces:          {summary['total_traces']}")
        print(f"  Total Events:          {summary['total_events']}")
        print(f"  Success Rate:          {summary['success_rate']:.1f}%")
        
        print(f"\n⏱️  PERFORMANCE:")
        print(f"  Avg Latency:           {summary['avg_latency_ms']:.2f} ms")
        print(f"  P95 Latency:           {summary['p95_latency_ms']:.2f} ms")
        
        print(f"\n🔢 TOKEN & COST:")
        print(f"  Total Tokens:          {summary['total_tokens']:,}")
        print(f"  Total Cost:            ${summary['total_cost_usd']:.4f}")
        
        print(f"\n🛡️  SAFETY & QUALITY:")
        print(f"  Guardrail Violations:  {summary['guardrail_violations']}")
        print(f"  Hallucinations:        {summary['hallucinations_detected']}")
        
        print("="*80 + "\n")
    
    def get_trace_tree(self, trace_id: str) -> str:
        events = self.collector.get_trace(trace_id)
        if not events:
            return "No trace found"
        
        tree_lines = []
        tree_lines.append(f"\nTrace: {trace_id}")
        tree_lines.append("="*80)
        
        for event in sorted(events, key=lambda e: e.timestamp):
            indent = "  " * self._get_depth(event, events)
            duration = event.metrics.get('latency_ms', 0)
            tree_lines.append(
                f"{indent}├─ [{event.event_type.value}] {event.name} "
                f"({duration:.2f}ms)"
            )
        
        return "\n".join(tree_lines)
    
    @staticmethod
    def _get_depth(event: TraceEvent, all_events: List[TraceEvent]) -> int:
        depth = 0
        parent_id = event.parent_id
        while parent_id:
            depth += 1
            parent = next((e for e in all_events if e.event_id == parent_id), None)
            parent_id = parent.parent_id if parent else None
        return depth
    
    def export_json(self, filepath: str):
        data = {
            "summary": self.collector.get_summary(),
            "traces": {
                trace_id: [e.to_dict() for e in events]
                for trace_id, events in self.collector.traces.items()
            },
            "guardrail_violations": [asdict(v) for v in self.collector.guardrail_violations],
            "hallucinations": [asdict(h) for h in self.collector.hallucinations],
            "anomalies": self.collector.anomalies
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported observability data to {filepath}")


class LangChainObserver:
    """Observer for LangChain/LangGraph integration"""
    
    def __init__(self):
        self.guardrails = GuardrailSystem()
        self.hallucination_detector = HallucinationDetector()
        self.anomaly_detector = AnomalyDetector()