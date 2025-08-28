"""
PDF generator for exporting reports and dashboards
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import json
import io
from pathlib import Path

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.platypus import PageBreak, Image, ListFlowable, ListItem
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.widgets.markers import makeMarker
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available. PDF generation will be disabled.")

from ..metrics.dora_metrics import DORAScorecard
from ..metrics.slo_metrics import SLODashboard
from ..verification.post_change_reports import PostChangeReport


logger = logging.getLogger(__name__)


class PDFGenerator:
    """PDF report generator for DevOps metrics and reports"""

    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation")

        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            alignment=1,  # Center
            spaceAfter=30,
            textColor=colors.darkblue
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=15,
            textColor=colors.darkgreen
        ))

        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=14,
            fontName='Helvetica-Bold',
            textColor=colors.black
        ))

        self.styles.add(ParagraphStyle(
            name='WarningText',
            parent=self.styles['Normal'],
            textColor=colors.red,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='SuccessText',
            parent=self.styles['Normal'],
            textColor=colors.green,
            fontName='Helvetica-Bold'
        ))

    def generate_dora_pdf(self, scorecard: DORAScorecard, output_path: Optional[str] = None) -> Union[bytes, str]:
        """
        Generate PDF report for DORA metrics scorecard

        Args:
            scorecard: DORA scorecard to export
            output_path: Optional file path to save PDF

        Returns:
            PDF content as bytes if no output_path, or file path if saved
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []

        # Title
        story.append(Paragraph(f"DORA Metrics Report", self.styles['ReportTitle']))
        story.append(Paragraph(f"Service: {scorecard.service_name}", self.styles['Heading2']))
        story.append(Paragraph(f"Period: {scorecard.time_period}", self.styles['Normal']))
        story.append(Paragraph(f"Generated: {scorecard.calculated_at.strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
        story.append(Spacer(1, 20))

        # Overall Rating
        story.append(Paragraph("Overall Performance Rating", self.styles['SectionHeader']))
        rating_color = self._get_rating_color(scorecard.overall_rating.value)
        story.append(Paragraph(f"<font color='{rating_color}'>{scorecard.overall_rating.value.upper()}</font>",
                              self.styles['MetricValue']))
        story.append(Spacer(1, 20))

        # Core Metrics Table
        story.append(Paragraph("Core DORA Metrics", self.styles['SectionHeader']))
        metrics_data = [
            ['Metric', 'Value', 'Target', 'Status'],
            ['Deployment Frequency', f"{scorecard.deployment_frequency:.2f}/day",
             "Multiple/day", self._get_performance_status(scorecard.deployment_frequency, 1/7)],
            ['Lead Time for Changes', f"{scorecard.lead_time_for_changes:.0f} min",
             "< 60 min", self._get_performance_status(60, scorecard.lead_time_for_changes, reverse=True)],
            ['Change Failure Rate', f"{scorecard.change_failure_rate:.1%}",
             "< 15%", self._get_performance_status(0.15, scorecard.change_failure_rate, reverse=True)],
            ['Time to Restore Service', f"{scorecard.time_to_restore_service:.0f} min",
             "< 60 min", self._get_performance_status(60, scorecard.time_to_restore_service, reverse=True)]
        ]

        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))

        # Additional Metrics
        story.append(Paragraph("Additional Metrics", self.styles['SectionHeader']))
        additional_data = [
            ['Metric', 'Value'],
            ['Deployment Success Rate', f"{scorecard.deployment_success_rate:.1%}"],
            ['Mean Time Between Failures', f"{scorecard.mean_time_between_failures:.0f} min"],
            ['Incident Count', str(scorecard.incident_count)],
            ['Deployment Count', str(scorecard.deployment_count)]
        ]

        additional_table = Table(additional_data, colWidths=[3*inch, 2*inch])
        additional_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(additional_table)
        story.append(Spacer(1, 20))

        # Trends
        if scorecard.trends:
            story.append(Paragraph("Performance Trends", self.styles['SectionHeader']))
            for period, metrics in scorecard.trends.items():
                story.append(Paragraph(f"{period} Period:", self.styles['Heading3']))
                trend_data = [
                    ['Metric', 'Value', 'Change %'],
                    ['Deployment Frequency', f"{metrics.get('deployment_frequency', 0):.2f}",
                     self._calculate_trend(scorecard.deployment_frequency, metrics.get('deployment_frequency', 0))],
                    ['Lead Time', f"{metrics.get('lead_time_for_changes', 0):.0f} min",
                     self._calculate_trend(scorecard.lead_time_for_changes, metrics.get('lead_time_for_changes', 0), reverse=True)],
                    ['Failure Rate', f"{metrics.get('change_failure_rate', 0):.1%}",
                     self._calculate_trend(scorecard.change_failure_rate, metrics.get('change_failure_rate', 0), reverse=True)],
                    ['Restore Time', f"{metrics.get('time_to_restore_service', 0):.0f} min",
                     self._calculate_trend(scorecard.time_to_restore_service, metrics.get('time_to_restore_service', 0), reverse=True)]
                ]

                trend_table = Table(trend_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
                trend_table.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)
                ]))
                story.append(trend_table)
                story.append(Spacer(1, 10))

        # Benchmarks
        if scorecard.benchmarks:
            story.append(Paragraph("Industry Benchmarks", self.styles['SectionHeader']))
            benchmark_data = [
                ['Level', 'Deploy Freq', 'Lead Time', 'Failure Rate', 'Restore Time'],
                ['Elite', 'Multiple/day', '< 1 hour', '< 15%', '< 1 hour'],
                ['High', 'Daily', '< 1 day', '< 30%', '< 1 day'],
                ['Medium', 'Weekly', '< 1 week', '< 45%', '< 1 week'],
                ['Low', 'Monthly+', '> 1 week', '> 45%', '> 1 week']
            ]

            benchmark_table = Table(benchmark_data, colWidths=[1*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
            benchmark_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (1, 1), (1, 1), colors.lightgreen)  # Highlight Elite row
            ]))
            story.append(benchmark_table)

        # Build PDF
        doc.build(story)

        if output_path:
            with open(output_path, 'wb') as f:
                f.write(buffer.getvalue())
            return output_path

        return buffer.getvalue()

    def generate_slo_pdf(self, dashboard: SLODashboard, output_path: Optional[str] = None) -> Union[bytes, str]:
        """
        Generate PDF report for SLO dashboard

        Args:
            dashboard: SLO dashboard to export
            output_path: Optional file path to save PDF

        Returns:
            PDF content as bytes if no output_path, or file path if saved
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []

        # Title
        story.append(Paragraph("SLO Dashboard Report", self.styles['ReportTitle']))
        story.append(Paragraph(f"Service: {dashboard.service_name}", self.styles['Heading2']))
        story.append(Paragraph(f"Period: {dashboard.time_period}", self.styles['Normal']))
        story.append(Paragraph(f"Generated: {dashboard.generated_at.strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
        story.append(Spacer(1, 20))

        # Overall Health
        story.append(Paragraph("Overall SLO Health", self.styles['SectionHeader']))
        health_color = self._get_slo_health_color(dashboard.overall_health.value)
        story.append(Paragraph(f"<font color='{health_color}'>{dashboard.overall_health.value.upper()}</font>",
                              self.styles['MetricValue']))
        story.append(Spacer(1, 20))

        # SLO Status Table
        if dashboard.slo_status:
            story.append(Paragraph("SLO Status Summary", self.styles['SectionHeader']))
            slo_data = [['SLO ID', 'Status']]
            for slo_id, status in dashboard.slo_status.items():
                status_color = self._get_slo_health_color(status.value)
                slo_data.append([slo_id, f"<font color='{status_color}'>{status.value.upper()}</font>"])

            slo_table = Table(slo_data, colWidths=[3*inch, 1.5*inch])
            slo_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(slo_table)
            story.append(Spacer(1, 20))

        # Error Budgets
        if dashboard.error_budgets:
            story.append(Paragraph("Error Budget Status", self.styles['SectionHeader']))
            budget_data = [['SLO ID', 'Budget Remaining %']]
            for slo_id, budget in dashboard.error_budgets.items():
                budget_color = colors.green if budget > 50 else colors.orange if budget > 10 else colors.red
                budget_data.append([slo_id, f"<font color='{budget_color}'>{budget:.1f}%</font>"])

            budget_table = Table(budget_data, colWidths=[3*inch, 1.5*inch])
            budget_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(budget_table)
            story.append(Spacer(1, 20))

        # Alerts
        if dashboard.alerts:
            story.append(Paragraph("Active Alerts", self.styles['SectionHeader']))
            for alert in dashboard.alerts:
                severity = alert.get('severity', 'info')
                alert_color = self._get_alert_color(severity)
                story.append(Paragraph(
                    f"<font color='{alert_color}'>[{severity.upper()}] {alert['message']}</font>",
                    self.styles['WarningText']
                ))
            story.append(Spacer(1, 20))

        # Recommendations
        if dashboard.recommendations:
            story.append(Paragraph("Recommendations", self.styles['SectionHeader']))
            for recommendation in dashboard.recommendations:
                story.append(Paragraph(f"• {recommendation}", self.styles['Normal']))
            story.append(Spacer(1, 20))

        # Build PDF
        doc.build(story)

        if output_path:
            with open(output_path, 'wb') as f:
                f.write(buffer.getvalue())
            return output_path

        return buffer.getvalue()

    def generate_post_change_pdf(self, report: PostChangeReport, output_path: Optional[str] = None) -> Union[bytes, str]:
        """
        Generate PDF report for post-change verification

        Args:
            report: Post-change report to export
            output_path: Optional file path to save PDF

        Returns:
            PDF content as bytes if no output_path, or file path if saved
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []

        # Title
        story.append(Paragraph("Post-Change Verification Report", self.styles['ReportTitle']))
        story.append(Paragraph(f"Incident: {report.incident_id}", self.styles['Heading2']))
        story.append(Paragraph(f"Fix: {report.fix_id}", self.styles['Normal']))
        story.append(Paragraph(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
        story.append(Spacer(1, 20))

        # Executive Summary
        if 'executive_summary' in report.sections:
            summary = report.sections['executive_summary']
            story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))

            success_status = "✅ SUCCESS" if report.overall_success else "❌ FAILURE"
            status_color = colors.green if report.overall_success else colors.red
            story.append(Paragraph(f"<font color='{status_color}'>{success_status}</font>",
                                  self.styles['MetricValue']))

            impact_desc = summary.get('impact_description', 'Unknown impact')
            story.append(Paragraph(f"Impact: {impact_desc}", self.styles['Normal']))

            story.append(Paragraph("Key Findings:", self.styles['Heading3']))
            for finding in summary.get('key_findings', []):
                story.append(Paragraph(f"• {finding}", self.styles['Normal']))

            story.append(Spacer(1, 20))

        # Incident Overview
        if 'incident_overview' in report.sections:
            incident = report.sections['incident_overview']
            story.append(Paragraph("Incident Overview", self.styles['SectionHeader']))

            incident_data = [
                ['Field', 'Value'],
                ['Title', incident.get('title', 'Unknown')],
                ['Severity', incident.get('severity', 'unknown')],
                ['Start Time', incident.get('start_time', 'unknown')],
                ['Duration', f"{incident.get('duration_minutes', 0)} minutes"],
                ['Affected Services', ', '.join(incident.get('affected_services', []))]
            ]

            incident_table = Table(incident_data, colWidths=[2*inch, 4*inch])
            incident_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)
            ]))
            story.append(incident_table)
            story.append(Spacer(1, 20))

        # Remediation Details
        if 'remediation_details' in report.sections:
            remediation = report.sections['remediation_details']
            story.append(Paragraph("Remediation Details", self.styles['SectionHeader']))

            remediation_data = [
                ['Field', 'Value'],
                ['Fix Name', remediation.get('fix_name', 'Unknown')],
                ['Fix Type', remediation.get('fix_type', 'unknown')],
                ['Risk Level', remediation.get('risk_level', 'unknown')],
                ['Applied At', remediation.get('applied_at', 'unknown')],
                ['Scripts Executed', str(remediation.get('scripts_executed', 0))]
            ]

            remediation_table = Table(remediation_data, colWidths=[2*inch, 4*inch])
            remediation_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)
            ]))
            story.append(remediation_table)
            story.append(Spacer(1, 20))

        # Canary Results
        if 'canary_results' in report.sections:
            canary = report.sections['canary_results']
            story.append(Paragraph("Canary Analysis Results", self.styles['SectionHeader']))

            decision = canary.get('decision', 'unknown')
            confidence = canary.get('confidence', 0)

            decision_color = colors.green if decision == 'success' else colors.red
            story.append(Paragraph(f"Decision: <font color='{decision_color}'>{decision.upper()}</font>",
                                  self.styles['MetricValue']))
            story.append(Paragraph(f"Confidence: {confidence:.1%}", self.styles['Normal']))

            if canary.get('improved_metrics'):
                story.append(Paragraph("Improved Metrics:", self.styles['Heading3']))
                for metric in canary['improved_metrics']:
                    story.append(Paragraph(f"• {metric}", self.styles['SuccessText']))

            if canary.get('degraded_metrics'):
                story.append(Paragraph("Degraded Metrics:", self.styles['Heading3']))
                for metric in canary['degraded_metrics']:
                    story.append(Paragraph(f"• {metric}", self.styles['WarningText']))

            story.append(Spacer(1, 20))

        # Recommendations
        if 'recommendations' in report.sections:
            recommendations = report.sections['recommendations']
            story.append(Paragraph("Recommendations", self.styles['SectionHeader']))

            for rec_type, recs in recommendations.items():
                if recs:
                    story.append(Paragraph(f"{rec_type.replace('_', ' ').title()}:", self.styles['Heading3']))
                    for rec in recs:
                        story.append(Paragraph(f"• {rec}", self.styles['Normal']))

            story.append(Spacer(1, 20))

        # Build PDF
        doc.build(story)

        if output_path:
            with open(output_path, 'wb') as f:
                f.write(buffer.getvalue())
            return output_path

        return buffer.getvalue()

    def _get_rating_color(self, rating: str) -> str:
        """Get color for DORA rating"""
        colors_map = {
            'elite': 'green',
            'high': 'blue',
            'medium': 'orange',
            'low': 'red'
        }
        return colors_map.get(rating.lower(), 'black')

    def _get_performance_status(self, value: float, target: float, reverse: bool = False) -> str:
        """Get performance status indicator"""
        if reverse:
            # Lower is better
            return "✅" if value <= target else "❌"
        else:
            # Higher is better
            return "✅" if value >= target else "❌"

    def _get_slo_health_color(self, status: str) -> str:
        """Get color for SLO health status"""
        colors_map = {
            'healthy': 'green',
            'warning': 'orange',
            'breached': 'red',
            'unknown': 'gray'
        }
        return colors_map.get(status.lower(), 'black')

    def _get_alert_color(self, severity: str) -> str:
        """Get color for alert severity"""
        colors_map = {
            'critical': 'red',
            'high': 'red',
            'medium': 'orange',
            'low': 'yellow',
            'info': 'blue'
        }
        return colors_map.get(severity.lower(), 'black')

    def _calculate_trend(self, current: float, previous: float, reverse: bool = False) -> str:
        """Calculate trend indicator"""
        if previous == 0:
            return "N/A"

        change = ((current - previous) / previous) * 100

        if reverse:
            # For metrics where lower is better
            if change < -10:
                return f"📈 {change:.1f}% (Improved)"
            elif change > 10:
                return f"📉 {change:.1f}% (Worsened)"
            else:
                return f"➡️ {change:.1f}% (Stable)"
        else:
            # For metrics where higher is better
            if change > 10:
                return f"📈 {change:.1f}% (Improved)"
            elif change < -10:
                return f"📉 {change:.1f}% (Worsened)"
            else:
                return f"➡️ {change:.1f}% (Stable)"
