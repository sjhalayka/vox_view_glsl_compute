#ifndef SHADER_UTILS_H
#define SHADER_UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Function to read shader source from file
std::string readShaderSource(const char* filePath) {
    std::string content;
    std::ifstream fileStream(filePath, std::ios::in);

    if (!fileStream.is_open()) {
        std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
        return "";
    }

    std::string line = "";
    while (!fileStream.eof()) {
        std::getline(fileStream, line);
        content.append(line + "\n");
    }

    fileStream.close();
    return content;
}

// Function to compile shader
GLuint compileShader(const char* source, GLenum shaderType) {
    GLuint shaderID = glCreateShader(shaderType);

    // Provide source code to shader
    glShaderSource(shaderID, 1, &source, NULL);

    // Compile shader
    glCompileShader(shaderID);

    // Check for errors
    GLint success = 0;
    glGetShaderiv(shaderID, GL_COMPILE_STATUS, &success);

    if (!success) {
        GLint logSize = 0;
        glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &logSize);

        std::vector<GLchar> errorLog(logSize);
        glGetShaderInfoLog(shaderID, logSize, &logSize, &errorLog[0]);

        std::string shaderTypeStr = shaderType == GL_VERTEX_SHADER ? "VERTEX" : "FRAGMENT";
        std::cerr << "ERROR: " << shaderTypeStr << " SHADER COMPILATION FAILED\n" << &errorLog[0] << std::endl;

        glDeleteShader(shaderID);
        return 0;
    }

    return shaderID;
}

// Function to create shader program
GLuint createShaderProgram(const char* vertexShaderSource, const char* fragmentShaderSource) {
    // Compile shaders
    GLuint vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);

    if (vertexShader == 0 || fragmentShader == 0) {
        return 0;
    }

    // Create program and attach shaders
    GLuint programID = glCreateProgram();
    glAttachShader(programID, vertexShader);
    glAttachShader(programID, fragmentShader);

    // Link program
    glLinkProgram(programID);

    // Check for errors
    GLint success = 0;
    glGetProgramiv(programID, GL_LINK_STATUS, &success);

    if (!success) {
        GLint logSize = 0;
        glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &logSize);

        std::vector<GLchar> errorLog(logSize);
        glGetProgramInfoLog(programID, logSize, &logSize, &errorLog[0]);

        std::cerr << "ERROR: SHADER PROGRAM LINKING FAILED\n" << &errorLog[0] << std::endl;

        glDeleteProgram(programID);
        return 0;
    }

    // Shaders are linked into the program, so we can delete them
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return programID;
}

#endif // SHADER_UTILS_H