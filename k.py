from graphviz import Digraph

# Create a directed graph
er = Digraph('ER_Diagram', filename='er_diagram', format='png')

# Define entities (tables)
er.node('Preprocessed Images', shape='box')
er.node('Classification Results', shape='box')

# Define attributes for Preprocessed Images
er.edge('Preprocessed Images', 'image_id', label="PK")
er.edge('Preprocessed Images', 'image_path')
er.edge('Preprocessed Images', 'category')
er.edge('Preprocessed Images', 'preprocessing_details')
er.edge('Preprocessed Images', 'upload_date')

# Define attributes for Classification Results
er.edge('Classification Results', 'result_id', label="PK")
er.edge('Classification Results', 'image_id', label="FK")  # Foreign Key linking to Preprocessed Images
er.edge('Classification Results', 'predicted_class')
er.edge('Classification Results', 'confidence_score')
er.edge('Classification Results', 'timestamp')

# Define Relationships
er.edge('Preprocessed Images', 'Classification Results', label="1:N (image_id)")

# Render and save the ER diagram
er.render(view=True)  # Opens the diagram automatically
