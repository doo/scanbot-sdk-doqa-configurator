import base64
import io
from pathlib import Path

import pandas as pd
import plotly.express as px
from PIL import Image
from sklearn.manifold import TSNE

"""
This function creates a 3D t-SNE plot of clustered data points with hoverable images.
It uses the TSNE algorithm from sklearn to reduce the dimensionality of the clustered data to 3D.
"""


def tsne_plot(training_dir: Path, pipeline, X: pd.DataFrame):
    clusters = pd.DataFrame(pipeline.named_steps['clustering'].transform(X))
    tsne = TSNE(n_components=3, random_state=0, perplexity=min(30, (len(clusters) - 1) // 3))
    projections = tsne.fit_transform(clusters)

    # Function to convert image to base64
    def image_to_base64(image_path):
        try:
            with Image.open(image_path) as img:
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return ""

    # Create a DataFrame with projections and image data
    plot_data = pd.DataFrame(
        {
            'x': projections[:, 0],
            'y': projections[:, 1],
            'z': projections[:, 2],
            'label': X['label'],
            'image_path': X['image_path'].astype(str),
        }
    )

    # Convert images to base64 for hover display
    print("Converting images to base64...")
    plot_data['image_b64'] = plot_data['image_path'].apply(image_to_base64)
    plot_data['filename'] = plot_data['image_path'].apply(lambda x: Path(x).name)

    # Create the 3D scatter plot
    fig = px.scatter_3d(
        plot_data,
        x='x',
        y='y',
        z='z',
        color='label',
        hover_data=['filename'],
        labels={'color': 'Label', 'x': 'X', 'y': 'Y', 'z': 'Z'},
    )

    fig.update_traces(marker_size=6)

    # Create a comprehensive HTML file with embedded images
    html_template = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>t-SNE Visualization with Image Hover</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            #hover-image {{
                position: absolute;
                pointer-events: none;
                z-index: 1000;
                background: white;
                border: 2px solid #333;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                display: none;
                max-width: 420px;
            }}
            #hover-image img {{
                max-width: 400px;
                max-height: 400px;
                display: block;
            }}
            body {{
                margin: 0;
                padding: 20px;
                font-family: Arial, sans-serif;
            }}
            #plotly-div {{
                width: 100%;
                height: 80vh;
            }}
        </style>
    </head>
    <body>
        <h1>t-SNE 3D Visualization with Image Hover</h1>
        <div id="plotly-div"></div>
        <div id="hover-image"></div>

        <script>
            // Image data embedded directly
            var imageData = {plot_data[['image_b64', 'filename']].to_json(orient='records')};

            // Create the plot using the same method as plotly express
            var trace = {{
                x: {plot_data['x'].tolist()},
                y: {plot_data['y'].tolist()},
                z: {plot_data['z'].tolist()},
                mode: 'markers',
                marker: {{
                    size: 6,
                    color: {plot_data['label'].tolist()},
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: {{
                        title: 'Label'
                    }}
                }},
                type: 'scatter3d',
                text: {plot_data['filename'].tolist()},
                hovertemplate: '<b>Label:</b> %{{marker.color}}<br>' +
                              '<b>X:</b> %{{x}}<br>' +
                              '<b>Y:</b> %{{y}}<br>' +
                              '<b>Z:</b> %{{z}}<br>' +
                              '<b>File:</b> %{{text}}<br>' +
                              '<extra></extra>'
            }};

            var layout = {{
                title: 't-SNE 3D Visualization',
                scene: {{
                    xaxis: {{title: 'X'}},
                    yaxis: {{title: 'Y'}},
                    zaxis: {{title: 'Z'}}
                }},
                margin: {{l: 0, r: 0, b: 0, t: 50}}
            }};

            // Create the plot
            Plotly.newPlot('plotly-div', [trace], layout, {{responsive: true}});

            // Get plot element
            var plotDiv = document.getElementById('plotly-div');
            var hoverDiv = document.getElementById('hover-image');

            // Add hover event listeners
            plotDiv.on('plotly_hover', function(eventData) {{
                console.log('Hover event:', eventData);
                console.log('Event points:', eventData.points);

                var point = eventData.points[0];
                console.log('Point object:', point);

                var pointIndex = point.pointIndex || point.pointNumber;

                console.log('Point index:', pointIndex, 'Total images:', imageData.length);

                if (pointIndex !== undefined && pointIndex < imageData.length && imageData[pointIndex] && imageData[pointIndex].image_b64) {{
                    var imgData = imageData[pointIndex];

                    hoverDiv.innerHTML =
                        '<img src="' + imgData.image_b64 + '">' +
                        '<div class="filename">' + imgData.filename + '</div>';

                    hoverDiv.style.display = 'block';

                    // Use mouse position from the plot container instead of event
                    var plotRect = plotDiv.getBoundingClientRect();
                    var mouseX = plotRect.left + plotRect.width * 0.8;  // Position near right edge
                    var mouseY = plotRect.top + 100;  // Position near top

                    hoverDiv.style.left = mouseX + 'px';
                    hoverDiv.style.top = mouseY + 'px';

                    console.log('Showing image for:', imgData.filename);
                }} else {{
                    console.log('No image data found for index:', pointIndex, 'Point object keys:', Object.keys(point));
                }}
            }});

            plotDiv.on('plotly_unhover', function() {{
                hoverDiv.style.display = 'none';
            }});

            console.log('Image hover setup complete. Total images:', imageData.length);
        </script>
    </body>
    </html>
            '''

    with open(training_dir / "tsne.html", 'w') as f:
        f.write(html_template)
