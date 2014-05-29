Img = imread('a.jpg');
Img= rgb2gray(Img);
Img=im2double(Img);
Var =0.001;

NImg=imnoise(Img,'gaussian',Var);

subplot(1,3,1);imshow(Img);title('Original Image');
subplot(1,3,2);imshow(NImg);title('Noisy Gaussian Image');

[r c] =size(Img);
EImg=zeros(r,c);
EImg(1,:) =Img(1,:);
EImg(:,1)=Img(:,1);


Pk =eye(3);
A=[0.35 0.35 0.3;0 1 0;0 0 1];
H=[1 0 0];

I=eye(3);
Q=0;
R=Var;

X=[EImg(2,1);EImg(1,2);EImg(1,1)];
for i=2:r
    for j=2:c
         
         X_pred = A*X;
         X_pred,X
         Pk=A*Pk*transpose(A)+Q;
         
         %kalman gain
         K=Pk*transpose(H)*inv(H*Pk*transpose(H)+R);
         
         %final update
         X=X_pred+K*(NImg(i,j)-H*X_pred);
         Pk=(I-K*H)*Pk;
         EImg(i,j)=X(1,1);
    end
end

subplot(1,3,3);imshow(EImg);title('Denoised Image');

