package com.example.model.api;

import java.io.IOException;

public interface ModelLoader {

    byte[] getModelData(String modelName) throws IOException;

}
