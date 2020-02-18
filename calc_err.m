function ERR = calc_err(XSO, H, epochs)
    [N, n] = size(XSO);

    limiter = floor(N * 0.7);

    for k = 1:epochs
        per = randperm(N);
        XSO = XSO(per, :);
        XS = XSO(1:limiter, :);
        Xe = XSO(1:limiter, :);

        % Mean and Covariance (FIT)
        xb = mean(XS);
        B  = cov(XS);

        % Estimation
        sig = 0.01;
        eobs = sig * randn(4, 1);
        y = H*(Xe') + eobs;
        % Posterior
        xa = (inv(B) + H' * (1 / sig^2) * H) \ (B \ (xb') + (1 / sig^2) * H' * y);
        % Covarianza posterior
        Ca = inv(inv(B) + H' * (1 / sig^2) * H);

        est9 = xa(5:8);
        ERR(k) = norm(est9 - Xe(5:8)', 1) / norm(Xe(5:8), 1);
    end
end