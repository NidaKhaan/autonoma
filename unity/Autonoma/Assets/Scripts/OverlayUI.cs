using UnityEngine;

public class OverlayUI : MonoBehaviour
{
    public EgoCarWaypointDriver wpDriver;
    public RaycastSensor        sensor;

    private Texture2D _solid;

    void Awake()
    {
        _solid = new Texture2D(1,1);
        _solid.SetPixel(0,0,Color.white);
        _solid.Apply();
    }

    void OnGUI()
    {
        if (wpDriver == null) return;

        float sw = Screen.width;
        float sh = Screen.height;

        // Panel — top left
        float pw = 280f, ph = 220f;
        float px = 12f,  py = 12f;

        // Background
        Box(px, py, pw, ph, new Color(0,0,0,0.65f));

        // Title bar
        Box(px, py, pw, 24f, new Color(0.1f,0.1f,0.15f,0.9f));
        Lbl(px+8, py+4, pw, 18, "AUTONOMA  —  AI DRIVE", 12,
            FontStyle.Bold, Color.white);

        float y = py + 30f;
        float lx = px + 8f;
        float bx = px + 100f;
        float bw = 130f;
        float vx = px + 238f;
        float rh = 22f;

        // Speed
        float spd = wpDriver.CurrentSpeed;
        Row(lx, y, bx, bw, vx, "Speed",
            spd / 80f, $"{spd:F0} km/h",
            new Color(0.3f,0.6f,1f)); y += rh;

        // Throttle
        Row(lx, y, bx, bw, vx, "Throttle",
            wpDriver.FinalThrottle, $"{wpDriver.FinalThrottle:F2}",
            new Color(0.2f,0.9f,0.3f)); y += rh;

        // Brake
        Row(lx, y, bx, bw, vx, "Brake",
            wpDriver.FinalBrake, $"{wpDriver.FinalBrake:F2}",
            new Color(1f,0.3f,0.3f)); y += rh;

        // Steer — centered
        RowCenter(lx, y, bx, bw, vx, "Steer",
            wpDriver.FinalSteer, $"{wpDriver.FinalSteer:F2}",
            new Color(1f,0.8f,0.1f)); y += rh;

        // Risk
        Color rc = wpDriver.CurrentRiskLevel == "CRITICAL" ? new Color(1f,0.2f,0.2f)
                 : wpDriver.CurrentRiskLevel == "CAUTION"  ? new Color(1f,0.7f,0f)
                 : new Color(0.2f,1f,0.4f);
        Row(lx, y, bx, bw, vx, "Risk",
            wpDriver.CurrentRisk/100f, $"{wpDriver.CurrentRisk:F0}",
            rc); y += rh;

        // Action
        Box(lx, y, pw-16f, 20f, new Color(0,0,0,0));
        Lbl(lx, y, 60, 20, "Action", 10, FontStyle.Normal, new Color(0.6f,0.6f,0.6f));
        Lbl(lx+62, y, 200, 20, wpDriver.CurrentAction, 12, FontStyle.Bold, rc);
        y += 24f;

        // Divider
        Box(lx, y, pw-16f, 1f, new Color(1,1,1,0.15f));
        y += 6f;

        // Raycast row
        if (sensor != null)
        {
            string ray = $"F:{sensor.frontDist:F1}  FL:{sensor.frontLeftDist:F1}" +
                         $"  FR:{sensor.frontRightDist:F1}  R:{sensor.backDist:F1}";
            Lbl(lx, y, pw-16f, 18, ray, 10, FontStyle.Normal, new Color(0.5f,0.5f,0.5f));
        }
    }

    void Row(float lx, float y, float bx, float bw, float vx,
             string label, float t, string val, Color c)
    {
        Lbl(lx, y, 90, 20, label, 10, FontStyle.Normal, new Color(0.65f,0.65f,0.65f));
        // Bar bg
        Box(bx, y+5, bw, 11, new Color(0.15f,0.15f,0.15f,0.9f));
        // Bar fill
        float fill = Mathf.Clamp01(t) * bw;
        if (fill > 0) Box(bx, y+5, fill, 11, c);
        Lbl(vx, y, 50, 20, val, 10, FontStyle.Bold, Color.white);
    }

    void RowCenter(float lx, float y, float bx, float bw, float vx,
                   string label, float t, string val, Color c)
    {
        Lbl(lx, y, 90, 20, label, 10, FontStyle.Normal, new Color(0.65f,0.65f,0.65f));
        Box(bx, y+5, bw, 11, new Color(0.15f,0.15f,0.15f,0.9f));
        float center = bx + bw * 0.5f;
        Box(center, y+3, 1, 15, new Color(0.4f,0.4f,0.4f));
        float fill = t * bw * 0.5f;
        if (fill > 0)       Box(center,        y+5, fill,  11, c);
        else if (fill < 0)  Box(center + fill, y+5, -fill, 11, c);
        Lbl(vx, y, 50, 20, val, 10, FontStyle.Bold, Color.white);
    }

    void Box(float x, float y, float w, float h, Color c)
    {
        GUI.color = c;
        GUI.DrawTexture(new Rect(x,y,w,h), _solid);
        GUI.color = Color.white;
    }

    void Lbl(float x, float y, float w, float h,
             string txt, int size, FontStyle fs, Color c)
    {
        var s = new GUIStyle(GUI.skin.label)
            { fontSize = size, fontStyle = fs };
        s.normal.textColor = c;
        GUI.Label(new Rect(x,y,w,h), txt, s);
    }
}