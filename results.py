import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

'''# Comfort explaining quantization - Professor, resources, explain
q_before = [5, 1, 0] # 0, 0, 1, 0, 0, 0
q_after = [1, 3, 2] # 0, 1, 1, 2, 2, 1

quant_comfort = pd.DataFrame({'Before': q_before, 'After': q_after})
fig_q = px.bar(quant_comfort, barmode='group', labels= {"index": "Student Comfort Explaining Quantization",
                                                        "value": "# of Respondents", "variable": "Legend"},
                                                        title="Student Comfort Explaining Quantization Before and After Workshop")
fig_q.update_xaxes(labelalias={0: "0 - Ask Professor", 1:"1 - Show Resources", 2:"2 - Explain"})
fig_q.update_layout(font_size=24)
#fig.show()

# Comfort explaining pruning
p_before = [6, 0, 0] # 0, 0, 0, 0, 0, 0
p_after = [1, 3, 2] # 1, 3, 2 0, 2, 1, 2, 1, 1

pruning_comfort = pd.DataFrame({'Before': p_before, 'After': p_after})
fig_p = px.bar(pruning_comfort, barmode='group', labels= {"index": "Student Comfort Explaining Pruning",
                                                        "value": "# of Respondents", "variable": "Legend"},
                                                        title="Student Comfort Explaining Pruning Before and After Workshop")
fig_p.update_layout(font_size=24)
fig_p.update_xaxes(labelalias={0: "0 - Ask Professor", 1:"1 - Show Resources", 2:"2 - Explain"})
#fig.show()

# Can you explain impact of pruning and quantization on a model
i_before = [4, 2, 0]
i_after = [0, 2, 4]

explain = pd.DataFrame({'Before': p_before, 'After': p_after})
fig_i = px.bar(explain, barmode='group', labels= {"index": "Can You Explain How Pruning and Quantization Impact a Model",
                                                        "value": "# of Respondents", "variable": "Legend"},
                                                        title="Can the Student Explain Pruning and Quantization Impacts on Model")
fig_i.update_layout(font_size=24)
fig_i.update_xaxes(labelalias={0: "No", 1:"Maybe", 2:"Yes"})
#fig.show()

# Put all of the graphs together
figures = [fig_q, fig_p, fig_i]
fig = make_subplots(rows=len(figures), cols=1) 

for i, figure in enumerate(figures):
    for trace in range(len(figure["data"])):
        fig.append_trace(figure["data"][trace], row=i+1, col=1)

fig.update_layout(font_size=24)
fig.update_xaxes(labelalias={0: "No", 1:"Maybe", 2:"Yes"})        
fig.show()'''

fig = make_subplots(rows=3, cols=1,
                    subplot_titles=("Student Comfort Explaining Quantization Before and After Workshop","Student Comfort Explaining Pruning Before and After Workshop", "Can You Explain How Pruning and Quantization Impact a Model"))

# Quantization
fig.add_trace(go.Bar(name="Before", x = ["0 - Ask Professor", "1 - Show Resources", "2 - Explain"], y = [6, 0, 0], marker_color="#ff9b68", showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(name="After", x = ["0 - Ask Professor", "1 - Show Resources", "2 - Explain"], y = [1, 3, 2], marker_color="#ff5600", showlegend=False), row=1, col=1)

# Pruning
fig.add_trace(go.Bar(name="Before", x = ["0 - Ask Professor", "1 - Show Resources", "2 - Explain"], y = [5, 1, 0], marker_color="#ff9b68", showlegend=False), row=2, col=1)
fig.add_trace(go.Bar(name="After", x = ["0 - Ask Professor", "1 - Show Resources", "2 - Explain"], y = [1, 3, 2], marker_color="#ff5600", showlegend=False), row=2, col=1)

# Explaining
fig.add_trace(go.Bar(name="Before", x = ["No", "Maybe", "Yes"], y = [4, 2, 0], marker_color="#ff9b68"), row=3, col=1)
fig.add_trace(go.Bar(name="After", x = ["No", "Maybe", "Yes"], y = [0, 2, 4], marker_color="#ff5600"), row=3, col=1)

# Update font sizes
fig.update_layout(font_size=24, font_family="Cambria", height=1200, width=1100)
for i in fig['layout']['annotations']:
    i['font'] = dict(size=24, family="Cambria")

fig.show()
#fig.write_image(file="results.svg")