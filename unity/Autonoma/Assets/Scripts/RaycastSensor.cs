using UnityEngine;

public class RaycastSensor : MonoBehaviour
{
    [Header("Ray Lengths")]
    public float frontRayLength      = 20f;
    public float frontAngleRayLength = 15f;
    public float sideRayLength       = 8f;
    public float backRayLength       = 10f;

    // Ignore the ego car's own colliders
    public LayerMask detectLayers = ~0;

    [HideInInspector] public float frontDist      = 20f;
    [HideInInspector] public float frontLeftDist  = 15f;
    [HideInInspector] public float frontRightDist = 15f;
    [HideInInspector] public float hardLeftDist   = 8f;
    [HideInInspector] public float hardRightDist  = 8f;
    [HideInInspector] public float leftLaneDist   = 8f;
    [HideInInspector] public float rightLaneDist  = 8f;
    [HideInInspector] public float backDist       = 10f;

    [HideInInspector] public bool redLightAhead    = false;
    [HideInInspector] public bool yellowLightAhead = false;
    [HideInInspector] public bool greenLightAhead  = false;
    [HideInInspector] public float frontObstacleDist = 20f;

    // Own colliders to ignore
    private Collider[] _ownColliders;

    // Heights for sweep — starts ABOVE bumper to skip own car body
    private static readonly float[] _sweepHeights = { 0.55f, 0.9f, 1.3f, 1.7f };

    void Awake()
    {
        _ownColliders = GetComponentsInChildren<Collider>();
    }

    void FixedUpdate()
    {
        frontDist      = Sweep(transform.forward,                            frontRayLength);
        frontLeftDist  = Sweep(Quaternion.Euler(0,-30,0)*transform.forward,  frontAngleRayLength);
        frontRightDist = Sweep(Quaternion.Euler(0, 30,0)*transform.forward,  frontAngleRayLength);
        hardLeftDist   = Sweep(Quaternion.Euler(0,-70,0)*transform.forward,  sideRayLength);
        hardRightDist  = Sweep(Quaternion.Euler(0, 70,0)*transform.forward,  sideRayLength);
        leftLaneDist   = Sweep(-transform.right,                             sideRayLength);
        rightLaneDist  = Sweep( transform.right,                             sideRayLength);
        backDist       = Sweep(-transform.forward,                           backRayLength);

        frontObstacleDist = Mathf.Min(frontDist, Mathf.Min(frontLeftDist, frontRightDist));

        DetectTrafficLight();
    }

    // Cast at multiple heights, skip own colliders, return closest hit
    float Sweep(Vector3 dir, float length)
    {
        float minDist = length;
        foreach (float h in _sweepHeights)
        {
            // Start origin 1.5m in front of center to skip own hood
            Vector3 origin = transform.position
                           + transform.forward * 1.5f
                           + Vector3.up * h;

            RaycastHit[] hits = Physics.RaycastAll(origin, dir, length, detectLayers);
            foreach (var hit in hits)
            {
                if (IsOwn(hit.collider)) continue;
                if (hit.distance < minDist)
                    minDist = hit.distance;
            }
        }
        return minDist;
    }

    bool IsOwn(Collider c)
    {
        foreach (var own in _ownColliders)
            if (own == c) return true;
        return false;
    }

    void DetectTrafficLight()
    {
        redLightAhead    = false;
        yellowLightAhead = false;
        greenLightAhead  = false;

        // ONLY straight ahead — not angled, not sides
        // Use a narrow cone: only pure forward at multiple heights
        float[] lightHeights = { 0.5f, 1.0f, 1.5f, 2.0f };
        foreach (float h in lightHeights)
        {
            // Start from car center (not offset) for traffic light detection
            Vector3 origin = transform.position + Vector3.up * h;
            RaycastHit[] hits = Physics.RaycastAll(
                origin, transform.forward, frontRayLength, detectLayers);
            foreach (var hit in hits)
            {
                if (IsOwn(hit.collider)) continue;
                if (hit.collider.name == "Stop")
                {
                    // Extra check: Stop collider must be roughly in front, not to the side
                    Vector3 toHit = (hit.point - transform.position).normalized;
                    float dot = Vector3.Dot(toHit, transform.forward);
                    if (dot > 0.85f)   // within ~32 degrees of forward only
                    {
                        redLightAhead = true;
                        return;
                    }
                }
            }
        }
    }

    // Called every Update frame from EgoCarWaypointDriver for instant reading
    public float InstantFrontDist(float maxDist = 22f)
    {
        return Sweep(transform.forward, maxDist);
    }
}