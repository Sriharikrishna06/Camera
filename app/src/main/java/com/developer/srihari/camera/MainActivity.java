package com.developer.srihari.camera;

import android.graphics.Bitmap;
import android.graphics.Camera;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import com.otaliastudios.cameraview.CameraListener;
import com.otaliastudios.cameraview.CameraUtils;
import com.otaliastudios.cameraview.CameraView;

import org.tensorflow.TensorFlow;

import java.util.Map;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    CameraView cameraView;
    TextView classlabels;
    ImageButton camera;
    TensorflowEngine inference;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        InferenceCache.classLabels=initMaps();
        cameraView=(CameraView)findViewById(R.id.cam_steam);
        cameraView.addCameraListener(cameraListener);
        cameraView.start();
        classlabels=findViewById(R.id.c_test);
        inference = new TensorflowEngine(this,"inception.pb");

    }

    CameraListener cameraListener=new CameraListener() {
        @Override
        public void onOrientationChanged(int orientation) {
            super.onOrientationChanged(orientation);
            //Toast.makeText(MainActivity.this, "Orientation Changed", Toast.LENGTH_SHORT).show();
        }

        @Override
        public void onPictureTaken(byte[] jpeg) {
            super.onPictureTaken(jpeg);
            Toast.makeText(MainActivity.this, "Picture Taken", Toast.LENGTH_SHORT).show();
            CameraUtils.decodeBitmap(jpeg,bitmapCallback);
        }
    };

    @Override
    public void onClick(View view) {

    }

    public void capture(View v){
        cameraView.capturePicture();
    }

    @Override
    protected void onStop() {
        super.onStop();
        cameraView.stop();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraView.destroy();
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraView.start();
    }

    @Override
    protected void onPause() {
        super.onPause();
        cameraView.stop();
    }

    private Map<Integer,String> initMaps() {
        try {
            ClassMapper reducer = new ClassMapper();
            MapLabels mapper = new MapLabels();
            Map<String, String> labels = mapper.loadLabels(
                    getAssets().open("labels.txt")
            );
            Map<Integer, String> nodes = mapper.loadNodes(
                    getAssets().open("labelmap.pbtxt")
            );
            return reducer.reduceMaps(nodes, labels);
        } catch (Exception e) {
            return null;
        }
    }

    CameraUtils.BitmapCallback bitmapCallback=new CameraUtils.BitmapCallback() {
        @Override
        public void onBitmapReady(Bitmap bitmap) {

            int prediction = inference.executeGraphOps("Mul","softmax",Bitmap.createScaledBitmap(bitmap,299,299,true));
            classlabels.setText(getLabel(prediction)+"");

        }
    };

    private String getLabel(int prediction) {
        return InferenceCache.classLabels.get(prediction);
    }
}
