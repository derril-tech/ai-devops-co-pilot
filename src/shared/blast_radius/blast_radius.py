"""
Blast radius computation using service topology graphs
"""
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class ServiceNode:
    """Represents a service in the topology"""
    id: str
    name: str
    type: str  # 'service', 'database', 'cache', 'queue', 'load_balancer'
    attributes: Dict[str, Any]
    health_score: float = 1.0
    criticality: float = 0.5  # 0-1 scale of business criticality


@dataclass
class ServiceEdge:
    """Represents a relationship between services"""
    source: str
    target: str
    relationship: str  # 'calls', 'depends_on', 'writes_to', 'reads_from'
    weight: float = 1.0  # Strength of relationship
    attributes: Dict[str, Any] = None


@dataclass
class BlastRadiusResult:
    """Result of blast radius calculation"""
    affected_services: List[ServiceNode]
    propagation_path: List[List[str]]  # Paths from root cause to affected services
    impact_score: float  # Overall impact (0-1)
    confidence: float  # Confidence in the calculation
    slo_impact: Dict[str, float]  # Impact on specific SLOs
    estimated_recovery_time: timedelta
    risk_factors: List[str]


class TopologyGraph:
    """Service topology graph for blast radius computation"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.services: Dict[str, ServiceNode] = {}
        self.edges: List[ServiceEdge] = []

    def add_service(self, service: ServiceNode):
        """Add a service to the topology"""
        self.services[service.id] = service
        self.graph.add_node(service.id, **service.__dict__)

    def add_edge(self, edge: ServiceEdge):
        """Add an edge between services"""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source,
            edge.target,
            relationship=edge.relationship,
            weight=edge.weight,
            **(edge.attributes or {})
        )

    def get_service(self, service_id: str) -> Optional[ServiceNode]:
        """Get service by ID"""
        return self.services.get(service_id)

    def get_neighbors(self, service_id: str, direction: str = 'out') -> List[ServiceNode]:
        """Get neighboring services"""
        if direction == 'out':
            neighbors = list(self.graph.successors(service_id))
        elif direction == 'in':
            neighbors = list(self.graph.predecessors(service_id))
        else:
            neighbors = list(self.graph.neighbors(service_id))

        return [self.services[nid] for nid in neighbors if nid in self.services]

    def get_shortest_paths(self, source: str, target: str) -> List[List[str]]:
        """Get shortest paths between services"""
        try:
            return list(nx.all_shortest_paths(self.graph, source, target))
        except nx.NetworkXNoPath:
            return []

    def calculate_centrality(self) -> Dict[str, float]:
        """Calculate service centrality scores"""
        # Use betweenness centrality as a measure of importance
        centrality = nx.betweenness_centrality(self.graph)

        # Normalize to 0-1 scale
        if centrality:
            max_centrality = max(centrality.values())
            if max_centrality > 0:
                centrality = {k: v / max_centrality for k, v in centrality.items()}

        return centrality

    def detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies"""
        cycles = list(nx.simple_cycles(self.graph))
        return cycles

    def get_service_layers(self) -> Dict[str, int]:
        """Assign services to layers based on dependency depth"""
        layers = {}

        # Find root services (no incoming edges)
        roots = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]

        # BFS to assign layers
        visited = set()
        queue = deque([(root, 0) for root in roots])

        while queue:
            service_id, layer = queue.popleft()

            if service_id in visited:
                continue

            visited.add(service_id)
            layers[service_id] = layer

            # Add successors to queue
            for successor in self.graph.successors(service_id):
                if successor not in visited:
                    queue.append((successor, layer + 1))

        return layers


class BlastRadiusCalculator:
    """Calculate blast radius of service failures"""

    def __init__(self, topology: TopologyGraph):
        self.topology = topology
        self.centrality_scores = topology.calculate_centrality()
        self.service_layers = topology.get_service_layers()

    def calculate_blast_radius(self, failed_service_id: str,
                              max_depth: int = 5) -> BlastRadiusResult:
        """
        Calculate blast radius from a failed service

        Args:
            failed_service_id: ID of the service that failed
            max_depth: Maximum depth to traverse in dependency graph

        Returns:
            BlastRadiusResult with affected services and impact assessment
        """
        affected_services = []
        propagation_paths = []
        visited = set()

        # BFS traversal from failed service
        queue = deque([(failed_service_id, [failed_service_id], 0)])

        while queue:
            current_service, path, depth = queue.popleft()

            if current_service in visited or depth > max_depth:
                continue

            visited.add(current_service)

            # Add to affected services (skip the root cause)
            if current_service != failed_service_id:
                if current_service in self.topology.services:
                    affected_services.append(self.topology.services[current_service])
                    propagation_paths.append(path)

            # Explore downstream dependencies
            for neighbor in self.topology.get_neighbors(current_service, 'out'):
                if neighbor.id not in visited:
                    new_path = path + [neighbor.id]
                    queue.append((neighbor.id, new_path, depth + 1))

        # Calculate impact metrics
        impact_score = self._calculate_impact_score(affected_services)
        confidence = self._calculate_confidence(affected_services)
        slo_impact = self._calculate_slo_impact(affected_services)
        recovery_time = self._estimate_recovery_time(affected_services)
        risk_factors = self._identify_risk_factors(failed_service_id, affected_services)

        return BlastRadiusResult(
            affected_services=affected_services,
            propagation_path=propagation_paths,
            impact_score=impact_score,
            confidence=confidence,
            slo_impact=slo_impact,
            estimated_recovery_time=recovery_time,
            risk_factors=risk_factors
        )

    def _calculate_impact_score(self, affected_services: List[ServiceNode]) -> float:
        """Calculate overall impact score"""
        if not affected_services:
            return 0.0

        # Factors contributing to impact
        service_count = len(affected_services)
        criticality_sum = sum(service.criticality for service in affected_services)
        centrality_sum = sum(self.centrality_scores.get(service.id, 0) for service in affected_services)

        # Normalize and combine factors
        count_factor = min(service_count / 10, 1.0)  # Cap at 10 services
        criticality_factor = criticality_sum / len(affected_services)
        centrality_factor = centrality_sum / len(affected_services)

        impact = (count_factor * 0.4 + criticality_factor * 0.4 + centrality_factor * 0.2)

        return min(impact, 1.0)

    def _calculate_confidence(self, affected_services: List[ServiceNode]) -> float:
        """Calculate confidence in the blast radius calculation"""
        if not affected_services:
            return 1.0  # High confidence for no impact

        # Confidence based on topology completeness and data quality
        base_confidence = 0.8  # Assume good baseline

        # Reduce confidence if we have incomplete topology
        missing_services = sum(1 for service in affected_services
                             if len(self.topology.get_neighbors(service.id)) == 0)
        missing_factor = 1.0 - (missing_services / len(affected_services) * 0.3)

        return base_confidence * missing_factor

    def _calculate_slo_impact(self, affected_services: List[ServiceNode]) -> Dict[str, float]:
        """Calculate impact on Service Level Objectives"""
        slo_impacts = {}

        for service in affected_services:
            # Extract SLO information from service attributes
            slo_targets = service.attributes.get('slo_targets', {})

            for slo_name, target in slo_targets.items():
                # Estimate impact based on service criticality and centrality
                base_impact = service.criticality * self.centrality_scores.get(service.id, 0.5)
                estimated_breach = min(base_impact * 2, 1.0)  # Conservative estimate

                if slo_name in slo_impacts:
                    slo_impacts[slo_name] = max(slo_impacts[slo_name], estimated_breach)
                else:
                    slo_impacts[slo_name] = estimated_breach

        return slo_impacts

    def _estimate_recovery_time(self, affected_services: List[ServiceNode]) -> timedelta:
        """Estimate time to recover from the incident"""
        if not affected_services:
            return timedelta(minutes=5)

        # Base recovery time
        base_time = timedelta(minutes=30)

        # Add time based on number of affected services
        service_factor = len(affected_services) * timedelta(minutes=15)

        # Add time based on criticality
        criticality_factor = timedelta(
            minutes=sum(service.criticality for service in affected_services) * 30
        )

        # Add time based on centrality (important services take longer to fix)
        centrality_factor = timedelta(
            minutes=sum(self.centrality_scores.get(service.id, 0) for service in affected_services) * 20
        )

        total_time = base_time + service_factor + criticality_factor + centrality_factor

        # Cap at reasonable maximum
        return min(total_time, timedelta(hours=24))

    def _identify_risk_factors(self, failed_service: str,
                             affected_services: List[ServiceNode]) -> List[str]:
        """Identify risk factors in the blast radius"""
        risk_factors = []

        # Check for critical services
        critical_services = [s for s in affected_services if s.criticality > 0.8]
        if critical_services:
            risk_factors.append(f"Affects {len(critical_services)} critical services")

        # Check for high centrality services
        high_centrality = [s for s in affected_services
                          if self.centrality_scores.get(s.id, 0) > 0.7]
        if high_centrality:
            risk_factors.append(f"Affects {len(high_centrality)} high-centrality services")

        # Check for cascading failures
        if len(affected_services) > 10:
            risk_factors.append("High risk of cascading failures")

        # Check for service layer violations
        failed_layer = self.service_layers.get(failed_service, 0)
        affected_layers = [self.service_layers.get(s.id, 0) for s in affected_services]

        if affected_layers and max(affected_layers) > failed_layer + 2:
            risk_factors.append("Deep dependency chain increases recovery complexity")

        # Check for circular dependencies
        cycles = self.topology.detect_cycles()
        if cycles:
            risk_factors.append(f"Detected {len(cycles)} circular dependencies")

        return risk_factors

    def analyze_failure_scenarios(self, service_id: str) -> Dict[str, BlastRadiusResult]:
        """Analyze different failure scenarios for a service"""
        scenarios = {}

        # Different failure depths
        for depth in [1, 3, 5, 10]:
            result = self.calculate_blast_radius(service_id, max_depth=depth)
            scenarios[f"depth_{depth}"] = result

        return scenarios

    def find_single_points_of_failure(self) -> List[Tuple[ServiceNode, BlastRadiusResult]]:
        """Identify services that would cause major disruptions if they fail"""
        critical_services = []

        for service_id, service in self.topology.services.items():
            # Only check high-criticality or high-centrality services
            if (service.criticality > 0.7 or
                self.centrality_scores.get(service_id, 0) > 0.7):

                blast_radius = self.calculate_blast_radius(service_id, max_depth=3)

                # Consider it a single point of failure if it affects many services
                if blast_radius.impact_score > 0.7:
                    critical_services.append((service, blast_radius))

        # Sort by impact score
        critical_services.sort(key=lambda x: x[1].impact_score, reverse=True)

        return critical_services

    def calculate_system_resilience(self) -> Dict[str, Any]:
        """Calculate overall system resilience metrics"""
        total_services = len(self.topology.services)
        if total_services == 0:
            return {"resilience_score": 0}

        # Find single points of failure
        spof = self.find_single_points_of_failure()

        # Calculate resilience score (inverse of SPOF impact)
        if spof:
            max_impact = max(result.impact_score for _, result in spof)
            resilience_score = 1.0 - max_impact
        else:
            resilience_score = 1.0

        # Calculate other metrics
        avg_centrality = np.mean(list(self.centrality_scores.values())) if self.centrality_scores else 0
        cycles = self.topology.detect_cycles()

        return {
            "resilience_score": resilience_score,
            "single_points_of_failure": len(spof),
            "average_centrality": avg_centrality,
            "circular_dependencies": len(cycles),
            "total_services": total_services,
            "connected_components": nx.number_weakly_connected_components(self.topology.graph)
        }
