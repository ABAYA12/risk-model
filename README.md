1. Risk Prediction
Objective: Predict risk scores and classify risks into categories (low/medium/high).
Prediction/Analysis	Tables Used	Key Columns Used	Graph Type	Caption
Risk Score Distribution	Risk	riskScore, riskCategory	Histogram	"Distribution of Risk Scores by Category (Low/Medium/High)"
Risk Probability vs. Impact	Risk	riskProbabilityLevel, riskImpactLevel	Scatter Plot	"Correlation Between Risk Probability and Impact Levels"
Risk Trends Over Time	Risk	createdAt, riskScore	Line Chart	"Risk Score Trends Over Time (Aggregated Monthly/Quarterly)"
2. Mitigation Suggestions
Objective: Suggest strategies based on risk type and effectiveness.
Prediction/Analysis	Tables Used	Key Columns Used	Graph Type	Caption
Mitigation Effectiveness	RiskMitigation	mitigatedRiskScore, mitigationControl	Bar Chart	"Impact of Mitigation Strategies on Reducing Risk Scores"
Cost vs. Effort of Mitigation	RiskMitigation	mitigationCost, mitigationEffort	Heatmap	"Cost vs. Effort of Mitigation Strategies (Low/Medium/High)"
Mitigation by Risk Category	Risk, RiskMitigation	riskCategory, mitigationControl	Stacked Bar Chart	"Most Common Mitigation Strategies by Risk Category"
3. Anomaly Detection
Objective: Detect unusual patterns in risk data.
Prediction/Analysis	Tables Used	Key Columns Used	Graph Type	Caption
Outlier Detection	Risk	riskScore, riskCategory	Box Plot	"Outliers in Risk Scores by Category"
Sudden Risk Spikes	Risk, RiskStatus	riskScore, statusType, createdAt	Time-Series Anomaly	"Unusual Risk Score Spikes Over Time (Anomalies Highlighted)"
4. Risk Monitoring
Objective: Track real-time risk status and responses.
Prediction/Analysis	Tables Used	Key Columns Used	Graph Type	Caption
Risk Status Distribution	RiskStatus	statusType, isMonitored	Pie Chart	"Proportion of Risks by Status (Monitored vs. Closed)"
Risk Response Activity	RiskMonitoring	riskResponseActivityStatus, comments	Gantt Chart	"Timeline of Risk Response Activities and Implementation Progress"
5. User Behavior Analysis
Objective: Identify risky user behaviors or gaps.
Prediction/Analysis	Tables Used	Key Columns Used	Graph Type	Caption
User Risk Submission Patterns	User, Risk	user.id, riskScore, createdAt	Line Chart	"Frequency of Risk Submissions by User Role/Department"
User Role vs. Risk Severity	User, UserRole, Risk	position, riskScore	Violin Plot	"Distribution of Risk Scores by User Role (e.g., Manager, Analyst)"