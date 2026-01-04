package com.example.model.api;

public interface Logger {

    void debug(String TAG, String text);
    void info(String TAG, String text);
    void error(String TAG, String text);

}
