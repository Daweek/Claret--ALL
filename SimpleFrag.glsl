#version 330 core

in vec4 fragmentColor;

// Ouput data
out vec3 color;

void main(){

	color = fragmentColor.xyz;

}
